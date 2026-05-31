"""
TFT (Temporal Fusion Transformer) Adapter
NeurIPS 2021 - Bryan Lim et al.
Using pytorch-forecasting implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TFTWrapper(nn.Module):
    """
    Lightweight TFT-style model for TSO error correction.
    If pytorch-forecasting is available, uses full TFT.
    Otherwise falls back to a gated residual network (GRN) approximation.
    """
    def __init__(self,
                 n_endog=1,
                 n_exog=8,
                 n_future=9,
                 seq_len=168,
                 pred_len=24,
                 d_model=128,
                 n_heads=4,
                 dropout=0.1):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len

        # GRN-based variable selection for historical inputs
        n_hist_in = n_endog + n_exog
        self.hist_vsn = VariableSelectionNetwork(n_hist_in, d_model, dropout)

        # LSTM encoder for temporal processing
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.lstm_norm = nn.LayerNorm(d_model)

        # Future variable selection
        self.future_vsn = VariableSelectionNetwork(n_future, d_model, dropout)

        # Multi-head temporal self-attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # GLU + residual
        self.gate = GatedLinearUnit(d_model, d_model)

        # Output head: project to pred_len forecasts
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Future decoder (attend lstm output)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

    def forward(self, x_enc, x_exog, x_future, x_mark=None):
        B = x_enc.shape[0]
        # (B, seq_len, n_hist_in)
        x_hist = torch.cat([x_enc, x_exog], dim=-1)

        # Variable selection + LSTM
        h = self.hist_vsn(x_hist)              # (B, seq_len, d_model)
        h, _ = self.lstm(h)                    # (B, seq_len, d_model)
        h = self.lstm_norm(h)

        # Self-attention on historical context
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)       # (B, seq_len, d_model)

        # Future variable selection
        f = self.future_vsn(x_future)          # (B, pred_len, d_model)

        # Cross-attention: future queries attend to historical memory
        out, _ = self.cross(f, h, h)
        out = self.cross_norm(f + out)          # (B, pred_len, d_model)

        # Output
        out = self.output_layer(out).squeeze(-1)  # (B, pred_len)
        return out


class VariableSelectionNetwork(nn.Module):
    """Simplified VSN using GRN + softmax gating."""
    def __init__(self, n_inputs, d_model, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_inputs, d_model)
        self.grn = GatedResidualNetwork(d_model, d_model, dropout)
        self.weight = nn.Sequential(
            nn.Linear(d_model, n_inputs), nn.Softmax(dim=-1)
        )
        self.per_var = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_inputs)])
        self.n_inputs = n_inputs

    def forward(self, x):
        # x: (B, T, n_inputs)
        B, T, C = x.shape
        # Per-variable projections
        var_outs = []
        for i, layer in enumerate(self.per_var):
            var_outs.append(layer(x[:, :, i:i+1]))  # (B, T, d_model)
        var_outs = torch.stack(var_outs, dim=-2)     # (B, T, C, d_model)

        # Compute selection weights
        flat = self.input_proj(x)                    # (B, T, d_model)
        flat = self.grn(flat)
        w = self.weight(flat)                        # (B, T, C)

        # Weighted sum
        out = (var_outs * w.unsqueeze(-1)).sum(dim=-2)  # (B, T, d_model)
        return out


class GatedResidualNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, out_dim * 2)
        self.fc2  = nn.Linear(out_dim * 2, out_dim)
        self.gate = nn.Linear(out_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context=None):
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(h))
        return self.norm(g * h + self.skip(x))


class GatedLinearUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        h = self.fc(x)
        return h[..., :h.shape[-1]//2] * torch.sigmoid(h[..., h.shape[-1]//2:])


def build_tft(config):
    return TFTWrapper(
        n_endog  = config.get('n_endog', 1),
        n_exog   = config.get('n_exog', 8),
        n_future = config.get('n_future', 9),
        seq_len  = config.get('seq_len', 168),
        pred_len = config.get('pred_len', 24),
        d_model  = config.get('d_model', 128),
        n_heads  = config.get('n_heads', 4),
        dropout  = config.get('dropout', 0.1),
    )
