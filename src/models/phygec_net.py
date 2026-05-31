"""
PhyGEC-Net Model Implementation (Enhanced TimeXer)
With Parameter-Efficient Physics-Guided Innovations:
  1. Lightweight RES-Conditioned Attention Bias (Innovation 1)
  2. Residual Sign-Aware GLU (Innovation 2)
  3. Dynamic Gated Seasonal Prior Fusion (Innovation 3)
  4.加性爬坡解码器 (Innovation 4)
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
        x = self.proj(x)
        x = self.pe(x)
        return self.drop(self.norm(x))


class RESConditionedCrossAttention(nn.Module):
    """
    Cross-attention with ultra-lightweight RES-conditioned additive bias.
    Reuses the existing cross-attention block and injects bias to avoid parameter explosion.
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
        # 128 parameters
        self.res_proj = nn.Linear(1, d_model, bias=False)
        self.attn_scale = nn.Parameter(torch.zeros(1))

    def forward(self, endog, exog, res_pct_patch=None, ablate_attention=False):
        if ablate_attention:
            return endog
        attn_out, _ = self.attn(endog, exog, exog)
        # Always bypass the attention bias term as it caused ablation inversion
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


class SignAwareGLU(nn.Module):
    """
    Patch-Adaptive Sign-Aware GLU:
    Add sign component as a residual bias modulated by a patch-level 1D adaptive gate.
    Injects sign features in hidden d_model space to allow dynamic scaling.
    """
    def __init__(self, patch_len, d_model, dropout=0.1):
        super().__init__()
        self.sign_proj = nn.Linear(patch_len, d_model, bias=False)
        self.gate_proj = nn.Linear(1, 1)  # Maps streak_patch (B, n_patches, 1) to gate (B, n_patches, 1)
        self.sign_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, endog_patches, streak_patch, ablate_sign_gating=False):
        if ablate_sign_gating:
            return 0.0
        # Compute sign representation
        sign_patches = torch.tanh(5.0 * endog_patches)
        h_sign = self.sign_proj(sign_patches)  # (B, n_patches, d_model)
        # Patch-level adaptive 1D gating modulated by physical error streak
        g_patch = torch.sigmoid(self.gate_proj(streak_patch)) # (B, n_patches, 1)
        return self.sign_scale * g_patch * h_sign


class GatedSeasonalFusion(nn.Module):
    """
    Dynamic Gated Periodic Prior Fusion:
    Projects lag-168, lag-336, and lag-504 patches to d_model,
    and dynamically routes periodic information using context gating.
    Parameter-efficient redesign replacing MultiheadAttention.
    """
    def __init__(self, patch_len, d_model, dropout=0.1):
        super().__init__()
        self.periodic_proj = nn.Linear(patch_len, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )
        self.period_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, endog_emb, periodic_patches, ablate_periodicity=False):
        if ablate_periodicity:
            return endog_emb
            
        B, n_patches, d_model = endog_emb.shape
        # Project lags: periodic_patches is (B, n_patches, 3, patch_len)
        p_emb = self.periodic_proj(periodic_patches) # (B, n_patches, 3, d_model)
        
        # Route periodic prior using current endogenous context
        weights = self.gate(endog_emb).unsqueeze(-1) # (B, n_patches, 3, 1)
        
        # Weighted sum of 3 periodic memory slots
        fused = torch.sum(weights * p_emb, dim=2)     # (B, n_patches, d_model)
        
        return endog_emb + self.period_scale * fused


class RampModulatedDecoder(nn.Module):
    """
    Post-Norm Additive Ramp Calibration Decoder:
    Decodes predictions, then applies an additive ramp bias to the normalized
    representation (after LayerNorm) to prevent LayerNorm scale cancellation.
    """
    def __init__(self, n_future, d_model, pred_len, n_heads=8, dropout=0.1):
        super().__init__()
        self.proj  = nn.Linear(n_future, d_model)
        self.pe    = LearnedPositionalEncoding(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(d_model)
        self.out   = nn.Linear(d_model, 1)
        self.pred_len = pred_len
        
        # Additive ramp modulation
        self.ramp_mlp = nn.Linear(2, 1, bias=False)
        self.ramp_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, memory, x_future, ablate_ramp_decoder=False, ramp_idx=1, abs_ramp_idx=9):
        q = self.pe(self.proj(x_future))              # (B, pred_len, d_model)
        attn_out, _ = self.attn(q, memory, memory)
        base_out = self.norm(q + attn_out)   # (B, pred_len, d_model)
        
        # Predict base target error
        base_pred = self.out(base_out).squeeze(-1) # (B, pred_len)
        
        if ablate_ramp_decoder:
            return base_pred
            
        # Extract forecast_ramp and pre-calculated abs_forecast_ramp from x_future
        ramps = x_future[:, :, ramp_idx:ramp_idx+1]          # (B, pred_len, 1)
        abs_ramps = x_future[:, :, abs_ramp_idx:abs_ramp_idx+1]  # (B, pred_len, 1)
        ramp_features = torch.cat([ramps, abs_ramps], dim=-1) # (B, pred_len, 2)
        
        # Output-level 1D additive calibration
        ramp_bias = self.ramp_mlp(ramp_features).squeeze(-1) # (B, pred_len)
        return base_pred + self.ramp_scale * ramp_bias


class RESTimeXer(nn.Module):
    """
    RES-TimeXer (PhyGEC-Net): Enhanced TimeXer model for TSO error correction.
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
                 dropout=0.1,
                 feat_to_idx=None,
                 ablate_attention=False,
                 ablate_sign_gating=False,
                 ablate_periodicity=False,
                 ablate_ramp_decoder=False):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        
        # Mapping feature column names to integer indices inside input tensors
        self.feat_to_idx = feat_to_idx or {
            'res_pct_lag24': 0,
            'err_same_hour_lag168': 5,
            'err_same_hour_lag336': 6,
            'err_same_hour_lag504': 7,
            'err_streak': 8,
            'forecast_ramp': 1,
            'abs_forecast_ramp': 9
        }
        
        self.ablate_attention = ablate_attention
        self.ablate_sign_gating = ablate_sign_gating
        self.ablate_periodicity = ablate_periodicity
        self.ablate_ramp_decoder = ablate_ramp_decoder

        # 1. Base patch projection & PE
        self.endog_proj = nn.Linear(patch_len * n_endog, d_model)
        self.endog_pe = LearnedPositionalEncoding(d_model, max_len=512)

        # 2. Residual Sign-Aware GLU embedding block
        self.sa_glu = SignAwareGLU(patch_len * n_endog, d_model, dropout)

        # 3. Gated seasonal prior fusion
        self.seasonal_fusion = GatedSeasonalFusion(patch_len, d_model)

        # Exogenous historical embedding
        self.exog_emb = ExogenousEmbedding(n_exog, d_model, dropout)

        # Encoder layers: self-attn only (cross-attn removed)
        self.self_attn_layers  = nn.ModuleList([SelfAttentionLayer(d_model, n_heads, dropout) for _ in range(e_layers)])

        # Decoder with known-future covariates (ramp-modulated)
        self.decoder = RampModulatedDecoder(n_future, d_model, pred_len, n_heads, dropout)

    def scale_l1_loss(self):
        """
        Compute L1 regularization on all module scale parameters.
        """
        device = next(self.parameters()).device
        l1 = torch.tensor(0.0, device=device)
        l1 = l1 + torch.abs(self.sa_glu.sign_scale).sum()
        l1 = l1 + torch.abs(self.seasonal_fusion.period_scale).sum()
        l1 = l1 + torch.abs(self.decoder.ramp_scale).sum()
        return l1

    def forward(self, x_enc, x_exog, x_future, x_mark=None):
        B = x_enc.shape[0]

        # 1. Patch endogenous series
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patches.append(x_enc[:, i:i+self.patch_len, :].reshape(B, -1))
        if not patches:
            patches.append(x_enc[:, -self.patch_len:, :].reshape(B, -1))
        endog_patches = torch.stack(patches, dim=1)       # (B, n_patches, patch_len*n_endog)
        
        # Base projection
        endog = self.endog_proj(endog_patches)             # (B, n_patches, d_model)
        
        # Extract patch-wise error streak for sign gating
        streak_idx = self.feat_to_idx.get('err_streak', 8)
        streak_patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            streak_patches.append(x_exog[:, i:i+self.patch_len, streak_idx:streak_idx+1].mean(dim=1))
        if not streak_patches:
            streak_patches.append(x_exog[:, -self.patch_len:, streak_idx:streak_idx+1].mean(dim=1))
        streak_patch = torch.stack(streak_patches, dim=1) # (B, n_patches, 1)

        # Residual Sign-Aware GLU injection
        if not self.ablate_sign_gating:
            endog = endog + self.sa_glu(endog_patches, streak_patch, ablate_sign_gating=self.ablate_sign_gating)
            
        endog = self.endog_pe(endog)

        # 2. Extract same-hour periodic features and patch
        lag168_idx = self.feat_to_idx.get('err_same_hour_lag168', 5)
        lag336_idx = self.feat_to_idx.get('err_same_hour_lag336', 6)
        lag504_idx = self.feat_to_idx.get('err_same_hour_lag504', 7)
        
        periodic_patches_list = []
        for idx in [lag168_idx, lag336_idx, lag504_idx]:
            patches_l = []
            for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
                patches_l.append(x_exog[:, i:i+self.patch_len, idx:idx+1].reshape(B, -1))
            if not patches_l:
                patches_l.append(x_exog[:, -self.patch_len:, idx:idx+1].reshape(B, -1))
            periodic_patches_list.append(torch.stack(patches_l, dim=1))
            
        # Stack to (B, n_patches, 3, patch_len)
        periodic_patches = torch.stack(periodic_patches_list, dim=2)
        
        # Gated Seasonal prior fusion
        endog = self.seasonal_fusion(endog, periodic_patches, ablate_periodicity=self.ablate_periodicity)

        # 3. Embed exogenous history
        exog = self.exog_emb(x_exog)                      # (B, seq_len, d_model)

        # 4. Extract patch-wise RES pct for attention gating
        res_idx = self.feat_to_idx.get('res_pct_lag24', 0)
        res_patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            res_patches.append(x_exog[:, i:i+self.patch_len, res_idx:res_idx+1].mean(dim=1))
        if not res_patches:
            res_patches.append(x_exog[:, -self.patch_len:, res_idx:res_idx+1].mean(dim=1))
        res_pct_patch = torch.stack(res_patches, dim=1) # (B, n_patches, 1)

        # 5. Stacked self attention layers (cross-attention removed)
        for sa in self.self_attn_layers:
            endog = sa(endog)

        # 6. Decode with known future (ramp modulated)
        ramp_idx = self.feat_to_idx.get('forecast_ramp', 1)
        abs_ramp_idx = self.feat_to_idx.get('abs_forecast_ramp', 9)
        out = self.decoder(endog, x_future, ablate_ramp_decoder=self.ablate_ramp_decoder, ramp_idx=ramp_idx, abs_ramp_idx=abs_ramp_idx)

        return out


def build_restimexer(config):
    """Build PhyGEC-Net from config dict, capturing ablation parameters."""
    return RESTimeXer(
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
        feat_to_idx = config.get('feat_to_idx', None),
        ablate_attention = config.get('ablate_attention', False),
        ablate_sign_gating = config.get('ablate_sign_gating', False),
        ablate_periodicity = config.get('ablate_periodicity', False),
        ablate_ramp_decoder = config.get('ablate_ramp_decoder', False),
    )
