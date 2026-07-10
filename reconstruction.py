import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1      = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu1      = nn.ReLU(inplace=True)
        self.conv2      = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu2      = nn.ReLU(inplace=True)
        self.conv3      = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out      = self.relu1(self.conv1(x))
        out      = self.relu2(self.conv2(out))
        out      = self.conv3(out)
        out      = out + identity

        return self.final_relu(out)


class ReconstructionBlock(nn.Module):
    def __init__(self, feat_ch: int = 64, out_ch: int = 13, n_blocks: int = 5):
        super().__init__()
        self.blocks     = nn.Sequential(*[ResidualBlock(feat_ch) for _ in range(n_blocks)])
        self.final_conv = nn.Conv2d(feat_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        
        return self.final_conv(x)
