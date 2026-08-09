import torch.nn as nn

from models import Autoformer, DLinear, iTransformer


class _PointwiseGate(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        d_model = int(getattr(configs, 'd_model', 32))
        dropout = float(getattr(configs, 'dropout', 0.1))
        self.net = nn.Sequential(
            nn.Linear(self.seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.pred_len * 2),
        )

    def forward(self, x_enc):
        # x_enc: (B, seq_len, N). Gate each channel history independently.
        bsz, _, n_channels = x_enc.shape
        x = x_enc.transpose(1, 2).contiguous()  # (B, N, seq_len)
        logits = self.net(x).view(bsz, n_channels, self.pred_len, 2)
        return logits.reshape(bsz * n_channels, self.pred_len, 2)


class _GatedBackbone(nn.Module):
    backbone_cls = None

    def __init__(self, configs):
        super().__init__()
        if self.backbone_cls is None:
            raise NotImplementedError('backbone_cls must be set in subclasses')
        self.backbone = self.backbone_cls(configs)
        self.gate = _PointwiseGate(configs)

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        return_aux_repr=False,
        **kwargs,
    ):
        out = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        if return_aux_repr:
            aux_logits = {'has_shipment': self.gate(x_enc)}
            return out, {'aux_logits': aux_logits}
        return out


class DLinearGate(_GatedBackbone):
    backbone_cls = DLinear.Model


class AutoformerGate(_GatedBackbone):
    backbone_cls = Autoformer.Model


class iTransformerGate(_GatedBackbone):
    backbone_cls = iTransformer.Model
