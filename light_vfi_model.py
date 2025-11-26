# light_vfi_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LightVFIUNet2D(nn.Module):
    """
    輕量版 VFI 模型：
    - input:  [B, in_frames*3, H, W]  (例如 4 幀 → 12 channels)
    - output: [B, 3, H, W]            (中間幀)
    """
    def __init__(self, in_frames: int = 4, base_channels: int = 32):
        super().__init__()
        in_ch = in_frames * 3
        ch = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_ch, ch)        # H,   W
        self.pool1 = nn.MaxPool2d(2)            # H/2, W/2

        self.enc2 = ConvBlock(ch, ch * 2)       # H/2, W/2
        self.pool2 = nn.MaxPool2d(2)            # H/4, W/4

        self.enc3 = ConvBlock(ch * 2, ch * 4)   # H/4, W/4
        self.pool3 = nn.MaxPool2d(2)            # H/8, W/8

        # Bottleneck
        self.bottleneck = ConvBlock(ch * 4, ch * 4)

        # Decoder
        self.dec3 = ConvBlock(ch * 4 + ch * 4, ch * 2)  # concat with enc3
        self.dec2 = ConvBlock(ch * 2 + ch * 2, ch)      # concat with enc2
        self.dec1 = ConvBlock(ch + ch, ch)              # concat with enc1

        self.out_conv = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_frames*3, H, W]
        e1 = self.enc1(x)             # [B, ch, H,   W]
        p1 = self.pool1(e1)           # [B, ch, H/2, W/2]

        e2 = self.enc2(p1)            # [B, 2ch, H/2, W/2]
        p2 = self.pool2(e2)           # [B, 2ch, H/4, W/4]

        e3 = self.enc3(p2)            # [B, 4ch, H/4, W/4]
        p3 = self.pool3(e3)           # [B, 4ch, H/8, W/8]

        b = self.bottleneck(p3)       # [B, 4ch, H/8, W/8]

        # Decoder
        d3 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)   # [B, 8ch, H/4, W/4]
        d3 = self.dec3(d3)               # [B, 2ch, H/4, W/4]

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)   # [B, 4ch, H/2, W/2]
        d2 = self.dec2(d2)               # [B, ch,  H/2, W/2]

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)   # [B, 2ch, H, W]
        d1 = self.dec1(d1)               # [B, ch,  H, W]

        out = self.out_conv(d1).sigmoid()  # [B,3,H,W], in [0,1]
        return out
