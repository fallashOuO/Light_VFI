# interpolate_light_vfi.py
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from light_vfi_model import LightVFIUNet2D
from lora_utils import add_lora_to_lightvfi, load_lora_state_dict_2d


@torch.no_grad()
def interpolate_video_light_vfi(
    input_video: str,
    output_video: str,
    base_ckpt: str = "",
    lora_ckpt: str = "",
    in_frames: int = 4,
    internal_h: int = 540,
    internal_w: int = 960,
    base_channels: int = 32,
    device: str = "cuda",
):
    # 裝置設定
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 讀取影片
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_bgr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()

    num_frames = len(frames_bgr)
    if num_frames < 2:
        raise RuntimeError("Video has < 2 frames, cannot interpolate.")

    print(f"[LightVFI] 影片幀數: {num_frames}, 原始解析度: {width}x{height}")
    print(f"[LightVFI] 內部推論解析度: {internal_w}x{internal_h}")

    # ================== 建立 Base 模型 ==================
    model = LightVFIUNet2D(in_frames=in_frames, base_channels=base_channels)

    if base_ckpt and Path(base_ckpt).is_file():
        state = torch.load(base_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"Loaded LightVFI base checkpoint from {base_ckpt}")
    else:
        print("⚠ 未載入 base_ckpt，使用隨機初始化參數（不建議正式使用）。")

    # ================== 是否注入 LoRA ==================
    if lora_ckpt and Path(lora_ckpt).is_file():
        print(f"[LightVFI] 注入 LoRA 結構並載入權重: {lora_ckpt}")
        # 先把模型的 Conv2d 包成 LoRAConv2d
        model = add_lora_to_lightvfi(model, r=8, lora_alpha=16, dropout=0.0)

        # 再載入 LoRA 的權重
        lora_state = torch.load(lora_ckpt, map_location="cpu")
        load_lora_state_dict_2d(model, lora_state)
        print(f"Loaded LightVFI LoRA checkpoint from {lora_ckpt}")
    else:
        print("[LightVFI] 未載入 LoRA，使用純 base 模型")

    # ==================================================
    model.to(device)
    model.eval()

    # 輸出影片 writer：fps*2，解析度為原始解析度
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps * 2.0, (width, height))

    # 先寫入第一幀
    out.write(frames_bgr[0])

    print(f"[LightVFI] 開始補幀，輸出幀數將為 {num_frames * 2 - 1} ...")

    for i in tqdm(range(num_frames - 1), desc="LightVFI Interpolating", ncols=100):
        # 取 [i-1, i, i+1, i+2]，邊界 clamp
        if in_frames == 4:
            idx0 = max(i - 1, 0)
            idx1 = i
            idx2 = i + 1
            idx3 = min(i + 2, num_frames - 1)
            idxs = [idx0, idx1, idx2, idx3]
        else:
            raise NotImplementedError("Only in_frames=4 implemented.")

        rgb_list = []
        for idx in idxs:
            bgr = frames_bgr[idx]
            # 1) resize 到 internal size
            small = cv2.resize(
                bgr,
                (internal_w, internal_h),
                interpolation=cv2.INTER_AREA,
            )
            # 2) BGR → RGB, 0-1
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # [3,h,w]
            rgb_list.append(rgb)

        frames_stack = torch.stack(rgb_list, dim=0)      # [4,3,h,w]
        b, t, c, h_int, w_int = 1, len(rgb_list), 3, internal_h, internal_w
        frames_stack = frames_stack.unsqueeze(0).to(device)  # [1,4,3,h,w]
        x = frames_stack.view(b, t * c, h_int, w_int)        # [1,12,h,w]

        # 推論中間幀
        pred = model(x)                  # [1,3,h,w]
        pred = pred.clamp(0.0, 1.0)[0]   # [3,h,w]
        pred_np = (pred.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        # 放大回原始解析度
        pred_up = cv2.resize(
            pred_np,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        pred_bgr = cv2.cvtColor(pred_up, cv2.COLOR_RGB2BGR)

        # 先寫入插值幀，再寫入下一原始幀
        out.write(pred_bgr)
        out.write(frames_bgr[i + 1])

    out.release()
    print(f"[LightVFI] 完成，輸出影片: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, default="")
    parser.add_argument("--lora_ckpt", type=str, default="")
    parser.add_argument("--in_frames", type=int, default=4)
    parser.add_argument("--internal_h", type=int, default=540)
    parser.add_argument("--internal_w", type=int, default=960)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    interpolate_video_light_vfi(
        input_video=args.input,
        output_video=args.output,
        base_ckpt=args.base_ckpt,
        lora_ckpt=args.lora_ckpt,
        in_frames=args.in_frames,
        internal_h=args.internal_h,
        internal_w=args.internal_w,
        base_channels=args.base_channels,
        device=args.device,
    )
