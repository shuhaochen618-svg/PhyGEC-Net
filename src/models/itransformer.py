"""
iTransformer Model Implementation
Paper: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
ICLR 2024, Yong Liu et al. (thuml group)
Adapted for TSO Load Forecast Error Correction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.timexer import LearnedPositionalEncoding


class iTransformerEncoder(nn.Module):
    """
    iTransformer: inverts the attention to variate-dimension instead of time.
    Each 'token' is an entire time series of a single variate.
    """
    def __init__(self, seq_len, d_model, n_heads, e_layers, n_variates, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_variates = n_variates

        # Variate embedding: embed each variate's full history to d_model
        self.variate_proj = nn.Linear(seq_len, d_model)
        self.variate_pe   = nn.Embedding(n_variates + 64, d_model)  # position per variate

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

    def forward(self, x):
        """
        x: (B, seq_len, n_variates)
        Returns: (B, n_variates, d_model)
        """
        B, L, C = x.shape
        # Transpose to variate-first: (B, C, L)
        x = x.permute(0, 2, 1)
        # Project each variate's time series
        x = self.variate_proj(x)   # (B, C, d_model)
        # Add variate positional encoding
        pos = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.variate_pe(pos)
        # Self-attention across variate dimension
        x = self.encoder(x)        # (B, C, d_model)
        return x


class iTransformer(nn.Module):
    """
    iTransformer for TSO error correction.
    Concatenates all features (endogenous + exogenous historical + future known)
    as separate variates, inverts attention to learn inter-variate correlations.
    """
    def __init__(self,
                 n_endog=1,
                 n_exog=8,
                 n_future=9,
                 seq_len=168,
                 pred_len=24,
                 d_model=256,
                 n_heads=8,
                 e_layers=4,
                 dropout=0.1):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len

        # Total variates in the historical window
        # Endogenous + exogenous historical = n_endog + n_exog variates
        n_hist_variates = n_endog + n_exog
        self.n_hist = n_hist_variates

        # iTransformer encoder on historical variates
        self.encoder = iTransformerEncoder(
            seq_len=seq_len, d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, n_variates=n_hist_variates, dropout=dropout
        )

        # Future known covariates encoder
        self.future_proj = nn.Linear(n_future, d_model)
        self.future_pe   = LearnedPositionalEncoding(d_model, max_len=pred_len + 1)

        # Cross-attention: future queries attend to historical variate memory
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x_enc, x_exog, x_future, x_mark=None):
        """
        x_enc:    (B, seq_len, n_endog)
        x_exog:   (B, seq_len, n_exog)
        x_future: (B, pred_len, n_future)
        """
        B = x_enc.shape[0]

        # Concatenate all historical variates: (B, seq_len, n_hist)
        x_hist = torch.cat([x_enc, x_exog], dim=-1)

        # iTransformer: attend across variates
        variate_mem = self.encoder(x_hist)  # (B, n_hist, d_model)

        # Future decoder
        q = self.future_pe(self.future_proj(x_future))   # (B, pred_len, d_model)
        cross_out, _ = self.cross_attn(q, variate_mem, variate_mem)
        out = self.norm_cross(q + cross_out)              # (B, pred_len, d_model)
        out = self.out_proj(out).squeeze(-1)              # (B, pred_len)

        return out


def build_itransformer(config):
    return iTransformer(
        n_endog  = config.get('n_endog', 1),
        n_exog   = config.get('n_exog', 8),
        n_future = config.get('n_future', 9),
        seq_len  = config.get('seq_len', 168),
        pred_len = config.get('pred_len', 24),
        d_model  = config.get('d_model', 256),
        n_heads  = config.get('n_heads', 8),
        e_layers = config.get('e_layers', 4),
        dropout  = config.get('dropout', 0.1),
    )
