"""
DADIGAN Discriminator. Fig. 7.

"The synthesized and real reference image are stacked as the input for
the discriminator." Read together with the loss L_cGAN(G, D) in Eq. (33)
-- D(m, n) for real pairs and D(m, G(m)) for fake pairs, where m is the
cloudy image -- the discriminator is conditional: it takes the
concatenation of the cloudy (conditioning) image and a candidate
cloud-free image (either the ground truth or the generator's output).

Architecture: five blocks of [Conv(4x4, stride 2) -> InstanceNorm ->
LeakyReLU], with output channel widths 2^(4+i) for i in [1, 5]
(i.e. 32, 64, 128, 256, 512), followed by one more convolution
(kernel 3, stride 1) that reduces to a single channel; the result is
globally pooled to the scalar "real/fake" score described in the paper
("the last convolutional layer outputs a 1D data").
"""

import torch
import torch.nn as nn


class DiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 4, stride: int = 2, norm: bool = True):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, c1: int = 13, c2: int = 2, in_extra: int = 0):
        """
        Args:
            c1: optical channels (used for both the condition and the candidate image)
            c2: SAR channels (used only as part of the conditioning input)
            in_extra: set to 0; kept for flexibility if additional conditioning
                channels are desired.
        """
        super().__init__()
        # condition = concat(I_o cloudy image, I_s SAR image); candidate = real/generated cloud-free image
        in_ch = c1 + c2 + c1 + in_extra

        widths = [2 ** (4 + i) for i in range(1, 6)]  # 32, 64, 128, 256, 512
        blocks = []
        prev_ch = in_ch
        for i, w in enumerate(widths):
            norm = i != 0  # no norm on the very first block, common GAN practice
            blocks.append(DiscBlock(prev_ch, w, kernel_size=4, stride=2, norm=norm))
            prev_ch = w
        self.blocks = nn.Sequential(*blocks)

        self.final_conv = nn.Conv2d(prev_ch, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, I_o: torch.Tensor, I_s: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        x = torch.cat([I_o, I_s, candidate], dim=1)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.pool(x).flatten(1)  # (B, 1) scalar realism score (logit)
        return x
