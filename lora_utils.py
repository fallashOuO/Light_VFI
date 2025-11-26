# lora_utils.py
import math
from typing import Iterable, Dict

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """
    把一個 Conv2d 包起來，做 LoRA:
    y = Conv(x) + B(A(x)) * scaling
    """
    def __init__(
        self,
        conv: nn.Conv2d,
        r: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = conv
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_ch = conv.in_channels
        out_ch = conv.out_channels
        kH, kW = conv.kernel_size

        self.lora_A = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(
            r,
            out_ch,
            kernel_size=(kH, kW),
            padding=conv.padding,
            bias=False,
        )

        # Init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # base conv 不訓練
        self.conv.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        dx = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return base + dx


def add_lora_to_lightvfi(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    dropout: float = 0.0,
    target_module_prefixes=("enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1"),
) -> nn.Module:
    """
    在 LightVFIUNet2D 的 Conv2d 上注入 LoRA。
    """
    for name, module in model.named_modules():
        if any(name.startswith(p) for p in target_module_prefixes):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Conv2d):
                    wrapped = LoRAConv2d(child, r=r, lora_alpha=lora_alpha, dropout=dropout)
                    setattr(module, child_name, wrapped)
    return model


def iter_lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """
    回傳所有 LoRAConv2d 的參數（A/B），方便 optimizer 只訓練這些。
    """
    for module in model.modules():
        if isinstance(module, LoRAConv2d):
            for p in module.lora_A.parameters():
                if p.requires_grad:
                    yield p
            for p in module.lora_B.parameters():
                if p.requires_grad:
                    yield p


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    把 model 中所有 LoRAConv2d 相關的權重抽出來成一個 state_dict。
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRAConv2d):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return lora_state


def load_lora_state_dict_2d(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """
    把 get_lora_state_dict 存出來的 dict 載回 model。
    """
    missing = []
    for name, module in model.named_modules():
        if isinstance(module, LoRAConv2d):
            key_A = f"{name}.lora_A.weight"
            key_B = f"{name}.lora_B.weight"
            if key_A in state_dict:
                module.lora_A.weight.data.copy_(state_dict[key_A])
            else:
                missing.append(key_A)
            if key_B in state_dict:
                module.lora_B.weight.data.copy_(state_dict[key_B])
            else:
                missing.append(key_B)
    if missing:
        print(f"[load_lora_state_dict_2d] Missing keys (ignore if expected): {missing}")
