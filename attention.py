"""
Progressive Dual Attention Fusion Module (PDAFM) for DADIGAN.

Implements:
    - CAB  : Cross-Attention Block                 -> Eqs. (21)-(26), Fig. 5
    - MSAB : Multi-head Self-Attention Block (+gate) -> Eqs. (27)-(31), Fig. 5
    - PDAFM: F = MSAB(CAB(CAB(P, S), V))            -> Eq. (4)

Implementation notes:
* The paper mentions "we used matrix factorization method to reduce the
  amount of calculation when calculating the attention weight" without
  giving the exact scheme. Full quadratic attention over a 256x256
  feature map (65536 tokens) is computationally prohibitive, so we
  approximate this with a spatial-reduction trick (average-pool the
  Key/Value branch by `reduction_ratio` before computing attention,
  similar to PVT's SRA). This is a documented, practical stand-in for
  their "matrix factorization" step and keeps the module runnable at
  full resolution.
* MSAB's Q/K/V projection "W" is described as a depth-wise convolution;
  we implement it with a 1x1 conv followed by a 3x3 depth-wise conv,
  which is the common realization of this idea in Restormer-style
  architectures that this paper's design is inspired by.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAB(nn.Module):
    """Cross-Attention Block. Fuses a `query` branch with a `key/value` branch.

    Eqs. (21)-(23):
        Attention = softmax(Q K^T / sqrt(d))
        F_M = Linear(V (1 - Attention)) + Q
        F_M = MLP(Norm(F_M)) + F_M
    """

    def __init__(self, dim: int = 64, reduction_ratio: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.scale = dim ** -0.5

        if reduction_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=reduction_ratio, stride=reduction_ratio)
        else:
            self.sr = None

    def forward(self, query_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = query_feat.shape

        Q = self.q_proj(query_feat).flatten(2).transpose(1, 2)  # B, N, C

        kv_src = kv_feat
        if self.sr is not None:
            kv_src = self.sr(kv_feat)
        K = self.k_proj(kv_src).flatten(2).transpose(1, 2)  # B, M, C
        V = self.v_proj(kv_src).flatten(2).transpose(1, 2)  # B, M, C

        attn = torch.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)  # B, N, M

        # If shapes of attn (N,M) and V (M,C) mismatch with (1 - attn) @ V we
        # simply proceed: (1 - attn) has the same shape as attn (N, M), and
        # matrix-multiplying with V (M, C) is valid regardless of M != N.
        comp = (1.0 - attn) @ V  # B, N, C

        fused = self.out_proj(comp) + Q
        fused = fused + self.mlp(self.norm(fused))

        fused = fused.transpose(1, 2).reshape(B, C, H, W)
        return fused


class MSAB(nn.Module):
    """Multi-head Self-Attention Block with a gating mechanism. Eqs. (27)-(31).

        Q, K, V = depthwise_conv(F)
        head_i  = softmax(Q_i K_i^T / sqrt(d)) V_i
        MSA(F)  = concat(head_1, ..., head_i)
        F_fused = MSA(F) (*) G(F) + F              ((*) = Hadamard product)
    """

    def __init__(self, dim: int = 64, num_heads: int = 8, reduction_ratio: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)

        def dw_proj():
            return nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            )

        self.q_proj = dw_proj()
        self.k_proj = dw_proj()
        self.v_proj = dw_proj()
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
        )

        if reduction_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=reduction_ratio, stride=reduction_ratio)
        else:
            self.sr = None

    def _to_heads(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, C, _, _ = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = x.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, heads, N, hd
        return x

    def forward(self, F_in: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F_in.shape

        F_norm = F_in.permute(0, 2, 3, 1)  # B,H,W,C
        F_norm = self.norm(F_norm).permute(0, 3, 1, 2)  # B,C,H,W

        Q = self.q_proj(F_norm)
        kv_src = self.sr(F_norm) if self.sr is not None else F_norm
        K = self.k_proj(kv_src)
        V = self.v_proj(kv_src)

        Qh = self._to_heads(Q, H, W)  # B, heads, N, hd
        Kh = self._to_heads(K, *kv_src.shape[-2:])
        Vh = self._to_heads(V, *kv_src.shape[-2:])

        attn = torch.softmax((Qh @ Kh.transpose(-2, -1)) * self.scale, dim=-1)  # B,heads,N,M
        out = attn @ Vh  # B, heads, N, hd

        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C).transpose(1, 2).reshape(B, C, H, W)
        msa_out = self.out_proj(out)

        gated = msa_out * self.gate(F_in)
        F_fused = gated + F_in
        return F_fused


class PDAFM(nn.Module):
    """Progressive Dual Attention Fusion Module. Eq. (4):
        F = MSAB( CAB( CAB(P, S), V ) )
    """

    def __init__(self, dim: int = 64, num_heads: int = 8, reduction_ratio: int = 8):
        super().__init__()
        self.cab1 = CAB(dim, reduction_ratio=reduction_ratio)
        self.cab2 = CAB(dim, reduction_ratio=reduction_ratio)
        self.msab = MSAB(dim, num_heads=num_heads, reduction_ratio=reduction_ratio)

    def forward(self, P: torch.Tensor, S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        F_M = self.cab1(P, S)
        F = self.cab2(F_M, V)
        F_fused = self.msab(F)
        return F_fused
