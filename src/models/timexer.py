"""
TimeXer Model Implementation
Paper: "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
NeurIPS 2024, Yuxuan Wang et al. (thuml group)
Adapted for TSO Load Forecast Error Correction task
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        return x + self.pe(pos)


class ExogenousEmbedding(nn.Module):
    """Project exogenous variables to d_model dimension."""
    def __init__(self, n_exog, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_exog, d_model)
        self.pe   = LearnedPositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, n_exog)
        x = self.proj(x)
        x = self.pe(x)
        return self.drop(self.norm(x))


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention: endogenous patches attend to exogenous tokens.
    Key innovation of TimeXer.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, endog, exog):
        # endog: (B, n_patches, d_model)  query
        # exog:  (B, n_exog_tokens, d_model)  key/value
        attn_out, _ = self.attn(endog, exog, exog)
        endog = self.norm1(endog + attn_out)
        endog = self.norm2(endog + self.ffn(endog))
        return endog


class SelfAttentionLayer(nn.Module):
    """Standard self-attention for endogenous patch representation."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class FutureKnownDecoder(nn.Module):
    """Incorporate known-future exogenous covariates into the forecast."""
    def __init__(self, n_future, d_model, pred_len, n_heads=8, dropout=0.1):
        super().__init__()
        self.proj  = nn.Linear(n_future, d_model)
        self.pe    = LearnedPositionalEncoding(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(d_model)
        self.out   = nn.Linear(d_model, 1)
        self.pred_len = pred_len

    def forward(self, memory, x_future):
        # memory: (B, n_patches, d_model)
        # x_future: (B, pred_len, n_future)
        q = self.pe(self.proj(x_future))    # (B, pred_len, d_model)
        attn_out, _ = self.attn(q, memory, memory)
        out = self.norm(q + attn_out)       # (B, pred_len, d_model)
        return self.out(out).squeeze(-1)    # (B, pred_len)


class TimeXer(nn.Module):
    """
    TimeXer: Transformer for time series with exogenous variables.
    Architecture:
      1. Patch embedding for endogenous series
      2. ExogenousEmbedding for historical exogenous
      3. Stacked (SelfAttention + CrossAttention) layers
      4. FutureKnownDecoder to incorporate known future covariates
    """
    def __init__(self,
                 n_endog=1,
                 n_exog=8,
                 n_future=9,
                 seq_len=168,
                 pred_len=24,
                 patch_len=24,
                 stride=12,
                 d_model=256,
                 n_heads=8,
                 e_layers=3,
                 dropout=0.1):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        # Endogenous patch embedding
        n_patches = max(1, (seq_len - patch_len) // stride + 1)
        self.endog_proj = nn.Linear(patch_len * n_endog, d_model)
        self.endog_pe   = LearnedPositionalEncoding(d_model, max_len=512)

        # Exogenous historical embedding
        self.exog_emb = ExogenousEmbedding(n_exog, d_model, dropout)

        # Encoder layers: self-attn on endogenous + cross-attn with exogenous
        self.self_attn_layers  = nn.ModuleList([SelfAttentionLayer(d_model, n_heads, dropout) for _ in range(e_layers)])
        self.cross_attn_layers = nn.ModuleList([CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(e_layers)])

        # Decoder with known-future covariates
        self.decoder = FutureKnownDecoder(n_future, d_model, pred_len, n_heads, dropout)

        self._n_patches = n_patches

    def forward(self, x_enc, x_exog, x_future, x_mark=None):
        """
        x_enc:    (B, seq_len, n_endog)    historical endogenous
        x_exog:   (B, seq_len, n_exog)     historical exogenous
        x_future: (B, pred_len, n_future)  known future covariates
        """
        B = x_enc.shape[0]

        # 1. Patch endogenous series
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patches.append(x_enc[:, i:i+self.patch_len, :].reshape(B, -1))
        if not patches:
            patches.append(x_enc[:, -self.patch_len:, :].reshape(B, -1))
        endog_patches = torch.stack(patches, dim=1)       # (B, n_patches, patch_len*n_endog)
        endog = self.endog_proj(endog_patches)             # (B, n_patches, d_model)
        endog = self.endog_pe(endog)

        # 2. Embed exogenous history
        exog = self.exog_emb(x_exog)                      # (B, seq_len, d_model)

        # 3. Stacked self+cross attention
        for sa, ca in zip(self.self_attn_layers, self.cross_attn_layers):
            endog = sa(endog)
            endog = ca(endog, exog)                        # cross-attend to exog

        # 4. Decode with known future
        out = self.decoder(endog, x_future)                # (B, pred_len)

        return out


def build_timexer(config):
    """Build TimeXer from config dict."""
    return TimeXer(
        n_endog  = config.get('n_endog', 1),
        n_exog   = config.get('n_exog', 8),
        n_future = config.get('n_future', 9),
        seq_len  = config.get('seq_len', 168),
        pred_len = config.get('pred_len', 24),
        patch_len= config.get('patch_len', 24),
        stride   = config.get('stride', 12),
        d_model  = config.get('d_model', 256),
        n_heads  = config.get('n_heads', 8),
        e_layers = config.get('e_layers', 3),
        dropout  = config.get('dropout', 0.1),
    )
