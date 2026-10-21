# src/models/light_vfi.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
    ) -> None:
        super().__init__()

        if in_ch <= 0:
            raise ValueError(
                f"in_ch must be greater than 0, got {in_ch}"
            )

        if out_ch <= 0:
            raise ValueError(
                f"out_ch must be greater than 0, got {out_ch}"
            )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class LightVFI(nn.Module):
    """
    Lightweight four-frame video interpolation model.

    Input:
        [batch_size, in_frames * 3, height, width]

    Default input:
        [batch_size, 12, height, width]

    Output:
        [batch_size, 3, height, width]

    Output range:
        [0, 1]
    """

    def __init__(
        self,
        in_frames: int = 4,
        base_ch: int = 32,
    ) -> None:
        super().__init__()

        if in_frames != 4:
            raise ValueError(
                "LightVFI currently requires exactly "
                f"4 input frames, got {in_frames}."
            )

        if base_ch <= 0:
            raise ValueError(
                "base_ch must be greater than 0, "
                f"got {base_ch}."
            )

        self.in_frames = in_frames
        self.base_ch = base_ch
        self.input_channels = in_frames * 3

        self.enc1 = ConvBlock(
            self.input_channels,
            base_ch,
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(
            base_ch,
            base_ch * 2,
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(
            base_ch * 2,
            base_ch * 4,
        )
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(
            base_ch * 4,
            base_ch * 4,
        )

        self.dec3 = ConvBlock(
            base_ch * 8,
            base_ch * 2,
        )

        self.dec2 = ConvBlock(
            base_ch * 4,
            base_ch,
        )

        self.dec1 = ConvBlock(
            base_ch * 2,
            base_ch,
        )

        self.out = nn.Conv2d(
            in_channels=base_ch,
            out_channels=3,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "LightVFI input must be a torch.Tensor."
            )

        if x.ndim != 4:
            raise ValueError(
                "LightVFI input must have shape "
                "[batch, channels, height, width], "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels "
                f"for {self.in_frames} RGB frames, "
                f"got {x.shape[1]}."
            )

        input_height = x.shape[-2]
        input_width = x.shape[-1]

        # 模型包含三次 2 倍下採樣。
        if input_height < 8 or input_width < 8:
            raise ValueError(
                "Input height and width must both be at least 8, "
                f"got {input_height}x{input_width}."
            )

        encoder_1 = self.enc1(x)

        encoder_2 = self.enc2(
            self.pool1(encoder_1)
        )

        encoder_3 = self.enc3(
            self.pool2(encoder_2)
        )

        bottleneck = self.bottleneck(
            self.pool3(encoder_3)
        )

        decoder_3 = F.interpolate(
            bottleneck,
            size=encoder_3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        decoder_3 = torch.cat(
            [
                decoder_3,
                encoder_3,
            ],
            dim=1,
        )

        decoder_3 = self.dec3(decoder_3)

        decoder_2 = F.interpolate(
            decoder_3,
            size=encoder_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        decoder_2 = torch.cat(
            [
                decoder_2,
                encoder_2,
            ],
            dim=1,
        )

        decoder_2 = self.dec2(decoder_2)

        decoder_1 = F.interpolate(
            decoder_2,
            size=encoder_1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        decoder_1 = torch.cat(
            [
                decoder_1,
                encoder_1,
            ],
            dim=1,
        )

        decoder_1 = self.dec1(decoder_1)

        output = self.out(decoder_1)

        return torch.sigmoid(output)