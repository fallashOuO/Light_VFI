import math
from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """
    LoRA-enhanced Conv2d：

    output = base_conv(x) + scaling * B(dropout(A(x)))

    A:
        1 × 1 convolution
        in_channels -> rank

    B:
        與原始 convolution 相同 kernel size
        rank -> out_channels

    scaling:
        alpha / rank
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if r <= 0:
            raise ValueError(
                f"LoRA rank must be greater than 0, got {r}"
            )

        if alpha <= 0:
            raise ValueError(
                f"LoRA alpha must be greater than 0, "
                f"got {alpha}"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "LoRA dropout must satisfy "
                f"0 <= dropout < 1, got {dropout}"
            )

        if not isinstance(conv, nn.Conv2d):
            raise TypeError(
                "LoRAConv2d requires an nn.Conv2d instance."
            )

        self.conv = conv
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        kernel_height, kernel_width = conv.kernel_size

        self.lora_A = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=r,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.lora_B = nn.Conv2d(
            in_channels=r,
            out_channels=conv.out_channels,
            kernel_size=(
                kernel_height,
                kernel_width,
            ),
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,
            bias=False,
            padding_mode=conv.padding_mode,
        )

        # A 使用一般初始化；B 從 0 開始。
        # 因此剛注入 LoRA 時，模型輸出與 Base Model 完全相同。
        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5),
        )

        nn.init.zeros_(
            self.lora_B.weight
        )

        self.dropout: nn.Module

        if dropout > 0.0:
            self.dropout = nn.Dropout2d(
                p=dropout
            )
        else:
            self.dropout = nn.Identity()

        # Base convolution 必須保持凍結。
        for parameter in self.conv.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        base_output = self.conv(x)

        lora_output = self.lora_A(x)
        lora_output = self.dropout(lora_output)
        lora_output = self.lora_B(lora_output)
        lora_output = lora_output * self.scaling

        if base_output.shape != lora_output.shape:
            raise RuntimeError(
                "Base and LoRA output shapes do not match: "
                f"base={tuple(base_output.shape)}, "
                f"lora={tuple(lora_output.shape)}"
            )

        return base_output + lora_output


def _get_submodule_by_name(
    model: nn.Module,
    module_name: str,
) -> nn.Module:
    if module_name == "":
        return model

    current_module = model

    for part in module_name.split("."):
        if not hasattr(current_module, part):
            raise AttributeError(
                f"Module '{module_name}' was not found."
            )

        current_module = getattr(
            current_module,
            part,
        )

        if not isinstance(current_module, nn.Module):
            raise TypeError(
                f"'{module_name}' is not an nn.Module."
            )

    return current_module


def _matches_target_prefix(
    module_name: str,
    target_prefixes: tuple[str, ...],
) -> bool:
    return any(
        module_name == prefix
        or module_name.startswith(prefix + ".")
        for prefix in target_prefixes
    )


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_prefixes: tuple[str, ...] = (
        "enc1",
        "enc2",
        "enc3",
        "bottleneck",
        "dec3",
        "dec2",
        "dec1",
    ),
) -> nn.Module:
    """
    將指定區域內的 Conv2d 替換為 LoRAConv2d。

    預設不對最後的輸出層 model.out 注入 LoRA。
    """
    if r <= 0:
        raise ValueError(
            f"LoRA rank must be greater than 0, got {r}"
        )

    if alpha <= 0:
        raise ValueError(
            f"LoRA alpha must be greater than 0, "
            f"got {alpha}"
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "LoRA dropout must satisfy "
            f"0 <= dropout < 1, got {dropout}"
        )

    if not target_prefixes:
        raise ValueError(
            "target_prefixes must not be empty."
        )

    existing_lora_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, LoRAConv2d)
    ]

    if existing_lora_modules:
        raise RuntimeError(
            "LoRA has already been injected into the model: "
            + ", ".join(existing_lora_modules)
        )

    targets: list[
        tuple[
            str,
            str,
            nn.Conv2d,
        ]
    ] = []

    # 先收集目標，避免遍歷途中修改模型結構。
    for module_name, module in model.named_modules():
        if not _matches_target_prefix(
            module_name,
            target_prefixes,
        ):
            continue

        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Conv2d):
                targets.append(
                    (
                        module_name,
                        child_name,
                        child_module,
                    )
                )

    if not targets:
        raise RuntimeError(
            "No Conv2d layers matched the requested "
            f"LoRA targets: {target_prefixes}"
        )

    # 先凍結完整 Base Model，包括沒有注入 LoRA 的輸出層。
    for parameter in model.parameters():
        parameter.requires_grad = False

    for (
        parent_name,
        child_name,
        original_conv,
    ) in targets:
        parent_module = _get_submodule_by_name(
            model,
            parent_name,
        )

        current_child = getattr(
            parent_module,
            child_name,
        )

        if current_child is not original_conv:
            raise RuntimeError(
                "Model structure changed during LoRA injection: "
                f"{parent_name}.{child_name}"
            )

        setattr(
            parent_module,
            child_name,
            LoRAConv2d(
                conv=original_conv,
                r=r,
                alpha=alpha,
                dropout=dropout,
            ),
        )

    injected_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, LoRAConv2d)
    ]

    print(
        f"[LoRA] injected_layers={len(injected_names)}"
    )

    for name in injected_names:
        print(f"  - {name}")

    return model


def lora_parameters(
    model: nn.Module,
) -> Iterable[nn.Parameter]:
    found_lora = False

    for module in model.modules():
        if isinstance(module, LoRAConv2d):
            found_lora = True
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()

    if not found_lora:
        raise RuntimeError(
            "No LoRA modules were found in the model."
        )


def get_lora_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not isinstance(module, LoRAConv2d):
            continue

        state_dict[
            f"{name}.lora_A.weight"
        ] = (
            module.lora_A.weight
            .detach()
            .cpu()
            .clone()
        )

        state_dict[
            f"{name}.lora_B.weight"
        ] = (
            module.lora_B.weight
            .detach()
            .cpu()
            .clone()
        )

    if not state_dict:
        raise RuntimeError(
            "No LoRA parameters were found when "
            "creating the checkpoint."
        )

    return state_dict


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    if not state_dict:
        raise ValueError(
            "The LoRA checkpoint is empty."
        )

    if not all(
        isinstance(name, str)
        and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise TypeError(
            "The LoRA checkpoint must map parameter "
            "names to tensors."
        )

    expected_keys: set[str] = set()
    lora_modules: list[
        tuple[str, LoRAConv2d]
    ] = []

    for name, module in model.named_modules():
        if not isinstance(module, LoRAConv2d):
            continue

        lora_modules.append(
            (
                name,
                module,
            )
        )

        expected_keys.add(
            f"{name}.lora_A.weight"
        )

        expected_keys.add(
            f"{name}.lora_B.weight"
        )

    if not lora_modules:
        raise RuntimeError(
            "No LoRA modules exist in the model. "
            "Call inject_lora() before loading a "
            "LoRA checkpoint."
        )

    checkpoint_keys = set(
        state_dict.keys()
    )

    missing_keys = sorted(
        expected_keys - checkpoint_keys
    )

    unexpected_keys = sorted(
        checkpoint_keys - expected_keys
    )

    if missing_keys:
        raise RuntimeError(
            "Missing LoRA checkpoint parameters: "
            + ", ".join(missing_keys)
        )

    if unexpected_keys:
        raise RuntimeError(
            "Unexpected LoRA checkpoint parameters: "
            + ", ".join(unexpected_keys)
        )

    with torch.no_grad():
        for name, module in lora_modules:
            key_a = f"{name}.lora_A.weight"
            key_b = f"{name}.lora_B.weight"

            weight_a = state_dict[key_a]
            weight_b = state_dict[key_b]

            if weight_a.shape != module.lora_A.weight.shape:
                raise RuntimeError(
                    f"Shape mismatch for {key_a}: "
                    f"checkpoint={tuple(weight_a.shape)}, "
                    f"model={tuple(module.lora_A.weight.shape)}"
                )

            if weight_b.shape != module.lora_B.weight.shape:
                raise RuntimeError(
                    f"Shape mismatch for {key_b}: "
                    f"checkpoint={tuple(weight_b.shape)}, "
                    f"model={tuple(module.lora_B.weight.shape)}"
                )

            module.lora_A.weight.copy_(
                weight_a.to(
                    device=module.lora_A.weight.device,
                    dtype=module.lora_A.weight.dtype,
                )
            )

            module.lora_B.weight.copy_(
                weight_b.to(
                    device=module.lora_B.weight.device,
                    dtype=module.lora_B.weight.dtype,
                )
            )

    print(
        f"[LoRA] loaded_parameters={len(expected_keys)}"
    )