import torch
import torch.nn as nn


class Model(nn.Module):
    """
    iTransformer (inverted Transformer) for multivariate forecasting.

    Minimal implementation integrated into this repo's unified interface:
      forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) -> (B, pred_len, c_out)

    Core idea:
      - Treat each variate/channel as a token.
      - Token embedding is a projection of the historical trajectory (length seq_len).
      - Self-attention models inter-variate dependencies.
      - Project back to future trajectory of length pred_len.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len) if self.task_name not in (
            'classification',
            'anomaly_detection',
            'imputation',
        ) else int(configs.seq_len)

        self.enc_in = int(getattr(configs, 'enc_in', getattr(configs, 'c_out', 1)))
        self.c_out = int(getattr(configs, 'c_out', self.enc_in))

        d_model = int(getattr(configs, 'd_model', 128))
        n_heads = int(getattr(configs, 'n_heads', 8))
        e_layers = int(getattr(configs, 'e_layers', 2))
        d_ff = int(getattr(configs, 'd_ff', 512))
        dropout = float(getattr(configs, 'dropout', 0.1))
        activation = str(getattr(configs, 'activation', 'gelu'))

        # Each channel token: (seq_len,) -> (d_model,)
        self.token_proj = nn.Linear(self.seq_len, d_model)
        self.token_norm = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)

        # Decode each channel token to future trajectory: (d_model,) -> (pred_len,)
        self.out_proj = nn.Linear(d_model, self.pred_len)

        # Optional: map channels if enc_in != c_out (rare in this repo)
        self.channel_map = None
        if self.enc_in != self.c_out:
            self.channel_map = nn.Linear(self.enc_in, self.c_out, bias=False)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc: (B, seq_len, enc_in)
        returns: (B, pred_len, c_out)
        """
        # Variate tokens: (B, N, T)
        x = x_enc.transpose(1, 2)
        # Tokenize by projecting each channel's history to d_model
        x = self.token_proj(x)
        x = self.token_norm(x)
        # Self-attention across variates/channels
        x = self.encoder(x)
        # Project to future steps for each channel: (B, N, pred_len)
        y = self.out_proj(x)
        # Back to (B, pred_len, N)
        y = y.transpose(1, 2)
        if self.channel_map is not None:
            y = self.channel_map(y)
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc)[:, -self.pred_len :, :]
        if self.task_name == 'imputation':
            # For simplicity, return reconstruction over input length
            return self.forecast(x_enc)
        if self.task_name == 'anomaly_detection':
            return self.forecast(x_enc)
        if self.task_name == 'classification':
            raise NotImplementedError('Classification is not implemented for iTransformer in this repo.')
        return None
