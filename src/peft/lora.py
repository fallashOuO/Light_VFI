import math
from typing import Iterable, Dict, List, Tuple

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """
    y = Conv(x) + B(A(x)) * (alpha / r)
    A: 1x1 conv (in -> r)
    B: kxk conv (r -> out)
    """
    def __init__(self, conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.conv = conv
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_ch = conv.in_channels
        out_ch = conv.out_channels
        kH, kW = conv.kernel_size

        self.lora_A = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(
            r,
            out_ch,
            kernel_size=(kH, kW),
            padding=conv.padding,
            stride=conv.stride,
            dilation=conv.dilation,
            groups=1,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # freeze base conv
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        dx = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return base + dx


def _get_submodule_by_name(model: nn.Module, module_name: str) -> nn.Module:
    if module_name == "":
        return model
    curr = model
    for part in module_name.split("."):
        curr = getattr(curr, part)
    return curr


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_prefixes=("enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1"),
) -> nn.Module:
    """
    先收集要替換的 Conv2d，再統一替換，避免一邊遍歷一邊改結構造成遞迴爆掉。
    """
    targets: List[Tuple[str, str, nn.Conv2d]] = []

    for module_name, module in model.named_modules():
        if not any(module_name.startswith(p) for p in target_prefixes):
            continue

        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                targets.append((module_name, child_name, child))

    for module_name, child_name, child in targets:
        parent = _get_submodule_by_name(model, module_name)
        # 避免重複注入
        if isinstance(getattr(parent, child_name), LoRAConv2d):
            continue
        setattr(parent, child_name, LoRAConv2d(child, r=r, alpha=alpha, dropout=dropout))

    return model


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            yield from m.lora_A.parameters()
            yield from m.lora_B.parameters()


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRAConv2d):
            sd[f"{name}.lora_A.weight"] = m.lora_A.weight.detach().cpu()
            sd[f"{name}.lora_B.weight"] = m.lora_B.weight.detach().cpu()
    return sd


def load_lora_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor]):
    for name, m in model.named_modules():
        if isinstance(m, LoRAConv2d):
            kA = f"{name}.lora_A.weight"
            kB = f"{name}.lora_B.weight"
            if kA in sd:
                m.lora_A.weight.data.copy_(sd[kA])
            if kB in sd:
                m.lora_B.weight.data.copy_(sd[kB])