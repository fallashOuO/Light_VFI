# export_light_onnx.py
import argparse
from pathlib import Path

import torch

from light_vfi_model import LightVFIUNet2D
from lora_utils import add_lora_to_lightvfi, load_lora_state_dict_2d


def export_light_onnx(
    base_ckpt: str,
    lora_ckpt: str,
    out_onnx: str,
    in_frames: int = 4,
    base_channels: int = 32,
    h: int = 360,
    w: int = 640,
    use_lora: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightVFIUNet2D(in_frames=in_frames, base_channels=base_channels)

    if base_ckpt and Path(base_ckpt).is_file():
        state = torch.load(base_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"Loaded base checkpoint: {base_ckpt}")
    else:
        print("⚠ 沒有載入 base_ckpt，ONNX 將使用隨機初始化參數。")

    # 注入 LoRA（如果有設定）
    if use_lora:
        model = add_lora_to_lightvfi(model, r=8, lora_alpha=16, dropout=0.0)
        if lora_ckpt and Path(lora_ckpt).is_file():
            lora_state = torch.load(lora_ckpt, map_location="cpu")
            load_lora_state_dict_2d(model, lora_state)
            print(f"Loaded LoRA checkpoint: {lora_ckpt}")
        else:
            print("⚠ 未載入 LoRA，仍以 base-only 匯出。")

    model.to(device)
    model.eval()

    dummy = torch.randn(1, in_frames * 3, h, w, device=device)
    out_onnx_path = Path(out_onnx)
    out_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy,
        out_onnx_path.as_posix(),
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        dynamic_axes=None,  # 一開始用固定解析度，避免 TensorRT 踩雷
    )

    print(f"[ONNX] Exported to {out_onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", type=str, default="")
    parser.add_argument("--lora_ckpt", type=str, default="")
    parser.add_argument("--out_onnx", type=str, default="onnx/light_vfi_360p.onnx")
    parser.add_argument("--in_frames", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--h", type=int, default=360)
    parser.add_argument("--w", type=int, default=640)
    parser.add_argument("--no_lora", action="store_true", help="不注入 LoRA，輸出純 base")
    args = parser.parse_args()

    export_light_onnx(
        base_ckpt=args.base_ckpt,
        lora_ckpt=args.lora_ckpt,
        out_onnx=args.out_onnx,
        in_frames=args.in_frames,
        base_channels=args.base_channels,
        h=args.h,
        w=args.w,
        use_lora=not args.no_lora,
    )
