"""Transformer-only structural control used in the ablation study.

The model receives numerical patches without a text prompt or reprogramming
layer and retains the event and magnitude output heads.
"""
import torch.nn as nn

from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MLPFlattenHead(nn.Module):
    """Nonlinear magnitude head used by the paper configuration."""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.fc1 = nn.Linear(nf, nf)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(head_dropout)
        self.fc2 = nn.Linear(nf, target_window)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = getattr(configs, 'patch_len', patch_len)
        self.stride = getattr(configs, 'stride', stride)
        n_heads = getattr(configs, 'n_heads', 8)
        encoder_layers = getattr(configs, 'transformer_encoder_layers', 4)
        dropout = getattr(configs, 'dropout', 0.1)

        if getattr(configs, 'use_multiscale_patch', False):
            from layers.Embed import MultiScalePatchEmbedding
            self.patch_embedding = MultiScalePatchEmbedding(
                configs.d_model, configs.seq_len, self.patch_len, self.stride, dropout,
                scales=getattr(configs, 'multiscale_patch_scales', None))
        else:
            self.patch_embedding = PatchEmbedding(
                configs.d_model, self.patch_len, self.stride, dropout)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # Match the prediction head dimension used by the main model.
        self.patch_proj = nn.Linear(configs.d_model, self.d_ff)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_ff,
            nhead=n_heads,
            dim_feedforward=self.d_ff * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        if getattr(configs, 'regression_head_mlp', False):
            self.output_projection = MLPFlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=dropout)
        else:
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=dropout)

        self.normalize_layers = Normalize(
            configs.enc_in, affine=False, non_norm=getattr(configs, 'no_revin', False))

        self.channel_mixing = getattr(configs, 'channel_mixing', False)
        if self.channel_mixing and configs.enc_in > 1:
            self.channel_mixer = nn.Linear(configs.enc_in, configs.enc_in)
        else:
            self.channel_mixer = None

        self.use_aux_loss = getattr(configs, 'use_aux_loss', False)
        if self.use_aux_loss:
            self.aux_has_shipment = nn.Linear(self.head_nf, self.pred_len * 2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_reprogramming_attention=False, return_aux_repr=False):
        if self.task_name != 'long_term_forecast' and self.task_name != 'short_term_forecast':
            return None
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        if self.channel_mixer is not None:
            x_enc = self.channel_mixer(x_enc)
        # Keep the channel dimension so PatchEmbedding can report n_vars=N.
        # It internally flattens B*N for the encoder; we restore B and N below
        # before the prediction head and RevIN denormalization.
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        if x_enc.is_cuda:
            target_dtype = next(self.patch_embedding.parameters()).dtype
            x_enc = x_enc.to(target_dtype)

        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.patch_proj(enc_out)
        enc_out = self.transformer_encoder(enc_out)

        dec_out = enc_out.reshape(
            -1, n_vars, enc_out.shape[-2], enc_out.shape[-1]
        )
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        block = dec_out

        aux_logits = None
        if return_aux_repr and self.use_aux_loss and hasattr(self, 'aux_has_shipment'):
            repr_flat = block.mean(dim=1).reshape(block.size(0), -1)
            aux_logits_flat = self.aux_has_shipment(repr_flat)
            aux_logits = {'has_shipment': aux_logits_flat.view(block.size(0), self.pred_len, 2)}

        dec_out = self.output_projection(block)
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')

        extra = {}
        if return_aux_repr and aux_logits is not None:
            extra['aux_logits'] = aux_logits
        if extra:
            return dec_out, extra
        return dec_out
