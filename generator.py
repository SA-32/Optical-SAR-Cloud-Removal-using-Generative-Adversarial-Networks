"""
DADIGAN Generator. Fig. 3.

Pipeline:
    Initialization -> Shared & Private Feature Extraction (DDIN) ->
    Feature Fusion (PDAFM) -> Reconstruction (RB) -> Result

Initialization (Fig. 3, left):
    P0 = Conv(I_o)
    S0 = Conv(Concat(I_o, I_s))
    V0 = Conv(I_s)
"""

import torch
import torch.nn as nn

from .modules import DDIN
from .attention import PDAFM
from .reconstruction import ReconstructionBlock


class Generator(nn.Module):
    def __init__(
        self,
        c1: int = 13,       # optical (multispectral) channels
        c2: int = 2,        # SAR channels (VV, VH)
        feat_ch: int = 64,  # internal feature width
        num_stages: int = 3,  # T, number of DDIN unfolding stages
        num_heads: int = 8,
        attn_reduction_ratio: int = 8,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        self.init_p = nn.Conv2d(c1, feat_ch, kernel_size=3, padding=1)
        self.init_v = nn.Conv2d(c2, feat_ch, kernel_size=3, padding=1)
        self.init_s = nn.Conv2d(c1 + c2, feat_ch, kernel_size=3, padding=1)

        self.ddin = DDIN(feat_ch=feat_ch, c1=c1, c2=c2, num_stages=num_stages)
        self.pdafm = PDAFM(dim=feat_ch, num_heads=num_heads, reduction_ratio=attn_reduction_ratio)
        self.rb = ReconstructionBlock(feat_ch=feat_ch, out_ch=c1)

    def forward(self, I_o: torch.Tensor, I_s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I_o: cloud-covered optical image, shape (B, c1, H, W)
            I_s: SAR image, shape (B, c2, H, W)
        Returns:
            Reconstructed cloud-free optical image, shape (B, c1, H, W)
        """
        P0 = self.init_p(I_o)
        V0 = self.init_v(I_s)
        S0 = self.init_s(torch.cat([I_o, I_s], dim=1))

        P_T, V_T, S_T = self.ddin(P0, V0, S0, I_o, I_s)
        fused = self.pdafm(P_T, S_T, V_T)
        result = self.rb(fused)
        return result
