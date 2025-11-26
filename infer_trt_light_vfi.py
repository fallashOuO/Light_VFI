# infer_trt_light_vfi.py
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import cuda.cudart as cudart  # 來自 cuda-python

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path: str):
    engine_path = Path(engine_path)
    if not engine_path.is_file():
        raise FileNotFoundError(f"找不到 TensorRT engine 檔案: {engine_path}")

    with engine_path.open("rb") as f:
        engine_bytes = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT engine 反序列化失敗")
    return engine


def allocate_buffers(engine, batch_size=1):
    """使用 cuda-python 分配 GPU 記憶體，回傳 bindings、host/device buffer."""
    n_bindings = engine.num_bindings
    bindings = [None] * n_bindings
    host_buffers = [None] * n_bindings
    device_buffers = [None] * n_bindings

    for i in range(n_bindings):
        binding_name = engine.get_binding_name(i)
        binding_shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))

        # Dynamic shape 的話，先不處理，後面 context.set_binding_shape 再設
        size = trt.volume(binding_shape) * batch_size
        host_buf = np.empty(size, dtype=dtype)

        # cudaMalloc
        err, dev_ptr = cudart.cudaMalloc(host_buf.nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc 失敗: {err}")

        bindings[i] = int(dev_ptr)
        host_buffers[i] = host_buf
        device_buffers[i] = dev_ptr

        print(f"[allocate] binding {i}: name={binding_name}, shape={binding_shape}, dtype={dtype}")

    return bindings, host_buffers, device_buffers


def do_inference(context, bindings, host_buffers, device_buffers, input_index, output_index, input_np: np.ndarray):
    """把 input_np 丟進指定 input binding，從指定 output binding 取結果。"""
    # input_np shape: [1, 12, H, W], float32
    host_buffers[input_index] = np.ascontiguousarray(input_np.astype(np.float32).ravel())

    # H2D
    err = cudart.cudaMemcpy(
        device_buffers[input_index],
        host_buffers[input_index].ctypes.data,
        host_buffers[input_index].nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy H2D 失敗: {err}")

    # 執行推論
    if not context.execute_v2(bindings):
        raise RuntimeError("TensorRT context.execute_v2 失敗")

    # D2H
    err = cudart.cudaMemcpy(
        host_buffers[output_index].ctypes.data,
        device_buffers[output_index],
        host_buffers[output_index].nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy D2H 失敗: {err}")

    # 還原 shape
    out_shape = context.get_binding_shape(output_index)
    out_np = host_buffers[output_index].reshape(out_shape)
    return out_np


def interpolate_video_trt(
    engine_path: str,
    input_video: str,
    output_video: str,
    internal_h: int,
    internal_w: int,
):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    # 這裡假設 engine 只有一個 input / 一個 output
    # input:  [1, 12, H, W]
    # output: [1,  3, H, W]
    input_index = engine.get_binding_index("input") if "input" in [engine.get_binding_name(i) for i in range(engine.num_bindings)] else 0
    output_index = 1 if input_index == 0 else 0

    # 若是 dynamic shape，設定一下
    input_shape = (1, 12, internal_h, internal_w)
    if engine.get_binding_shape(input_index).num_dims == -1 or -1 in engine.get_binding_shape(input_index):
        context.set_binding_shape(input_index, input_shape)

    # 依照目前 binding shape 分配 buffer
    bindings, host_buffers, device_buffers = allocate_buffers(engine, batch_size=1)

    # 讀影片
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {input_video}")

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

    if len(frames_bgr) < 2:
        raise RuntimeError("影片幀數少於 2，無法補幀")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps * 2.0, (width, height))

    num_frames = len(frames_bgr)
    print(f"[TRT] 讀到 {num_frames} 幀，開始 30->60fps 補幀")

    # 先寫入第一幀
    out.write(frames_bgr[0])

    for i in range(num_frames - 1):
        # 用四幀 [i-1, i, i+1, i+2]（邊界 clamp）
        idx0 = max(i - 1, 0)
        idx1 = i
        idx2 = i + 1
        idx3 = min(i + 2, num_frames - 1)
        idxs = [idx0, idx1, idx2, idx3]

        frames_12ch = []
        for idx in idxs:
            bgr = frames_bgr[idx]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (internal_w, internal_h), interpolation=cv2.INTER_AREA)
            rgb = rgb.astype(np.float32) / 255.0  # [H,W,3]
            chw = np.transpose(rgb, (2, 0, 1))    # [3,H,W]
            frames_12ch.append(chw)

        stack_12 = np.concatenate(frames_12ch, axis=0)  # [12,H,W]
        stack_12 = np.expand_dims(stack_12, axis=0)     # [1,12,H,W]

        # TRT 推論
        pred = do_inference(context, bindings, host_buffers, device_buffers, input_index, output_index, stack_12)
        # pred: [1,3,H,W]
        pred = np.clip(pred, 0.0, 1.0)
        pred = pred[0]  # [3,H,W]
        pred = np.transpose(pred, (1, 2, 0))  # [H,W,3]
        pred = (pred * 255.0).astype(np.uint8)
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_CUBIC)
        pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

        # 寫入插值幀 + 原本下一幀
        out.write(pred_bgr)
        out.write(frames_bgr[i + 1])

        if (i + 1) % 50 == 0:
            print(f"[TRT] 處理到 {i+1}/{num_frames-1} 段")

    out.release()
    print(f"[TRT] 完成，輸出到 {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, required=True, help="TensorRT engine 檔案 (.engine)")
    parser.add_argument("--input", type=str, required=True, help="輸入影片 (30fps)")
    parser.add_argument("--output", type=str, required=True, help="輸出影片 (60fps)")
    parser.add_argument("--internal_h", type=int, default=360, help="內部推論高度 (例如 360)")
    parser.add_argument("--internal_w", type=int, default=640, help="內部推論寬度 (例如 640)")
    args = parser.parse_args()

    interpolate_video_trt(
        engine_path=args.engine,
        input_video=args.input,
        output_video=args.output,
        internal_h=args.internal_h,
        internal_w=args.internal_w,
    )
