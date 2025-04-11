import torch
import torch.nn as nn
import torch.nn.functional as F

class PeLU(nn.Module):
    def __init__(self):
        super(PeLU, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.elu(x) * self.alpha

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.conv(w)
        return x * w

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            PeLU(),
            CALayer(in_channels)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            PeLU(),
            CALayer(in_channels)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            PeLU(),
            CALayer(in_channels)
        )
        self.merge = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        merged = torch.cat([b1, b2, b3], dim=1)
        out = self.merge(merged)
        return out + x  # short skip

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(FeatureExtractor, self).__init__()
        layers = [ResidualBlock(in_channels) for _ in range(num_blocks)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale):
        super(UpsampleBlock, self).__init__()
        if scale == 1.5:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=1.5, mode='bicubic', align_corners=False),
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * scale * scale, 3, padding=1),
                nn.PixelShuffle(scale)
            )

    def forward(self, x):
        return self.up(x)

class TherISuRNet(nn.Module):
    def __init__(self, scale):
        super(TherISuRNet, self).__init__()
        self.scale = scale
        self.lfe = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            PeLU()
        )
        self.hfe1 = nn.Sequential(
            FeatureExtractor(64, num_blocks=4),
            UpsampleBlock(64, 2)
        )
        self.hfe2 = nn.Sequential(
            FeatureExtractor(64, num_blocks=2),
            UpsampleBlock(64, 2 if scale == 4 else 1.5 if scale == 3 else 1)
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        self.grl = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False),
            nn.Conv2d(1, 64, kernel_size=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x_lfe = self.lfe(x)
        x_hfe1 = self.hfe1(x_lfe)
        x_hfe2 = self.hfe2(x_hfe1)
        x_rec = self.reconstruction(x_hfe2)
        x_grl = self.grl(x)
        return x_rec + x_grl
