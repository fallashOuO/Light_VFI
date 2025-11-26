# Light_VFI
Lightweight Video Frame Interpolation (VFI) with Base Model + LoRA + ONNX + TensorRT

Light_VFI 是一個輕量級影片補幀系統，包含完整訓練、推論、加速與即時顯示流程。

特色：
- 輕量化 UNet2D 補幀模型（4-frame input → 中間幀）
- LoRA 微調（多場景快速適配）
- PyTorch 離線推論
- PyTorch 即時顯示（Live Demo）
- ONNX / TensorRT 加速推論
- 可做 30fps → 60fps 補幀

---

## 1. 專案架構

Light_VFI/
    datasets.py
    train_light_vfi.py
    train_light_lora.py
    interpolate_light_vfi.py
    interpolate_light_vfi_live.py
    infer_trt_light_vfi.py
    light_vfi_model.py
    lora_utils.py
    export_light_onnx.py
    extract_frames.py
    README.md
    requirements.txt

---

## 2. 安裝環境

pip install -r requirements.txt

建議環境：
- Python 3.10 ~ 3.11
- PyTorch + CUDA
- NVIDIA GPU
- 可選：TensorRT

---

## 3. Dataset 格式

frames/
    video1/
        000001.png
        000002.png
        ...
    video2/
        ...

每 4 張 frame → 1 個 sample。

---

## 4. Base Model 訓練

python train_light_vfi.py --frames_root frames --save_dir weights_light_base --epochs 10 --device cuda

輸出：
weights_light_base/light_vfi_base_final.pth

---

## 5. LoRA 微調（選用）

python train_light_lora.py \
  --frames_root frames \
  --base_ckpt weights_light_base/light_vfi_base_final.pth \
  --save_dir weights_light_lora \
  --device cuda

---

## 6. 推論（PyTorch版本）

### Base-only 推論：
python interpolate_light_vfi.py \
  --input videos_train/input.mp4 \
  --output outputs/output_interp.mp4 \
  --base_ckpt weights_light_base/light_vfi_base_final.pth \
  --device cuda

### Base + LoRA 推論：
python interpolate_light_vfi.py \
  --input input.mp4 \
  --output output_lora.mp4 \
  --base_ckpt weights_light_base/light_vfi_base_final.pth \
  --lora_ckpt weights_light_lora/light_vfi_lora_final.pth \
  --device cuda

---

## 7. Live Demo（即時顯示）

python interpolate_light_vfi_live.py \
  --input videos_train/input.mp4 \
  --base_ckpt weights_light_base/light_vfi_base_final.pth \
  --device cuda

此模式會彈出視窗，左邊原始幀、右邊插值幀。

---

## 8. 匯出 ONNX（用於 TensorRT）

python export_light_onnx.py \
  --base_ckpt weights_light_base/light_vfi_base_final.pth \
  --out_onnx onnx/light_vfi.onnx \
  --h 544 --w 960 \
  --no_lora

---

## 9. TensorRT 推論

### 轉 engine：
trtexec --onnx=onnx/light_vfi.onnx --saveEngine=trt/light_vfi_fp16.engine --fp16

### 推論：
python infer_trt_light_vfi.py \
  --engine trt/light_vfi_fp16.engine \
  --input input.mp4 \
  --output output_trt.mp4 \
  --internal_h 544 \
  --internal_w 960

---

## 10. 模型架構說明（LightVFI UNet2D）

- Encoder × 3
- Bottleneck
- Decoder × 3
- Skip Connection
- Input: 4 frames（堆疊成 12 channels）
- Output: 中間幀 (3 channels)

此架構為小模型，即時性好，訓練快速。

---

## 11. LoRA 支援

LightVFI 提供：
- Conv2D → LoRAConv2D
- 僅訓練 LoRA 權重
- 可快速切換不同場景設定

適合多場景影片補幀微調。

---

## 12. 適用領域

- 影片補幀（30→60fps）
- 動畫補幀
- 即時影像平滑（Live Video）
- 遊戲畫面補幀（類 DLSS Frame Generation）
- 學術研究 / 作業 / 專題展示

---

## 13. 引用

Light_VFI: Lightweight Video Frame Interpolation System (2025)  
Author: fallashOuO

---

## 14. 未來可以加入的改善方向

- 加入 Optical Flow 估計
- 支援多倍率補幀（×4、×8）
- 增加 Transformer 模組
- TensorRT INT8 加速
- 即時 OBS/遊戲畫面補幀（Real-Time Screen Capture）

# Light_VFI
A lightweight Video Frame Interpolation (VFI) system featuring:
- A compact UNet-based base model  
- LoRA fine-tuning for multi-scene adaptation  
- PyTorch offline and live (real-time display) inference  
- ONNX export and TensorRT acceleration  

This project is suitable for:
- Graduate-level research projects  
- Video enhancement and frame interpolation experiments  
- Studies on parameter-efficient fine-tuning (LoRA)  
- Real-time or near real-time frame generation on GPU  

---

## 1. Project Structure

Light_VFI/  
    datasets.py  
    train_light_vfi.py  
    train_light_lora.py  
    interpolate_light_vfi.py  
    interpolate_light_vfi_live.py  
    infer_trt_light_vfi.py  
    light_vfi_model.py  
    lora_utils.py  
    export_light_onnx.py  
    extract_frames.py  
    requirements.txt  
    README.md  

(Training data, weights, ONNX models, TensorRT engines, videos, and virtual environments are intentionally excluded via `.gitignore`.)

---

## 2. Installation

Install dependencies:

    pip install -r requirements.txt

Recommended environment:

- Python 3.10–3.11  
- PyTorch with CUDA enabled  
- NVIDIA GPU (e.g., RTX 40/50 series)  
- Optional: TensorRT for high-speed inference  

---

## 3. Dataset Format (Frame Folder)

Training data is organized as frame folders:

    frames/
        video1/
            000001.png
            000002.png
            ...
        video2/
            000001.png
            000002.png
            ...

Each video folder contains consecutive frames.  
The training dataset script will construct samples from consecutive frame windows (e.g., 4 input frames → 1 target frame).

You can use `extract_frames.py` to automatically extract frames from raw videos.

---

## 4. Base Model Training

Train the lightweight UNet-based VFI model:

    python train_light_vfi.py ^
        --frames_root frames ^
        --save_dir weights_light_base ^
        --epochs 10 ^
        --device cuda

After training, the final base model checkpoint is saved as:

    weights_light_base/light_vfi_base_final.pth

This checkpoint is used for both direct inference and as the base for LoRA fine-tuning.

---

## 5. LoRA Fine-Tuning (Optional)

LoRA (Low-Rank Adaptation) is used to adapt the base model to new scenes or domains by only training a small number of additional parameters while freezing the original weights.

Run LoRA training:

    python train_light_lora.py ^
        --frames_root frames ^
        --base_ckpt weights_light_base/light_vfi_base_final.pth ^
        --save_dir weights_light_lora ^
        --device cuda

This will produce a LoRA-only checkpoint, for example:

    weights_light_lora/light_vfi_lora_final.pth

Advantages of LoRA in this project:

- Fast adaptation to new video styles or scenes  
- Very small additional storage cost  
- Multiple LoRA modules can be swapped without retraining the base model  

---

## 6. Inference (PyTorch)

### 6.1 Base Model Inference

Run interpolation using only the base model:

    python interpolate_light_vfi.py ^
        --input videos_train/input.mp4 ^
        --output outputs/output_interp_base.mp4 ^
        --base_ckpt weights_light_base/light_vfi_base_final.pth ^
        --device cuda

The script reads the input video, interpolates intermediate frames, and writes the output video.  
By default, the output FPS is doubled (e.g., 30 → 60 FPS).

### 6.2 Base + LoRA Inference

If you have a trained LoRA checkpoint:

    python interpolate_light_vfi.py ^
        --input videos_train/input.mp4 ^
        --output outputs/output_interp_lora.mp4 ^
        --base_ckpt weights_light_base/light_vfi_base_final.pth ^
        --lora_ckpt weights_light_lora/light_vfi_lora_final.pth ^
        --device cuda

The base model weights are loaded first, then LoRA layers are injected and their weights are loaded on top.

---

## 7. Live Demo (Real-Time Visualization)

For presentation and qualitative evaluation, a live visualization script is provided.  
It performs frame interpolation and immediately displays the result using OpenCV, rather than writing to disk.

Run:

    python interpolate_light_vfi_live.py ^
        --input videos_train/input.mp4 ^
        --base_ckpt weights_light_base/light_vfi_base_final.pth ^
        --device cuda

A window will appear showing:

- Left: original frame  
- Right: interpolated frame  

This is useful when demonstrating model behavior to advisors, committees, or in-class presentations.

---

## 8. Export to ONNX

The base model (and optionally LoRA-augmented model) can be exported to ONNX for deployment and acceleration.

Example: export base model at internal resolution 960×544:

    python export_light_onnx.py ^
        --base_ckpt weights_light_base/light_vfi_base_final.pth ^
        --out_onnx onnx/light_vfi_base_544x960.onnx ^
        --in_frames 4 ^
        --h 544 ^
        --w 960 ^
        --no_lora

Notes:

- Internal height/width are chosen so that they are divisible by 8, matching the UNet downsampling depth.  
- The exported ONNX model is then used by TensorRT or other runtimes.

---

## 9. TensorRT Acceleration

### 9.1 Convert ONNX to TensorRT Engine

Using `trtexec` (from NVIDIA TensorRT), convert the ONNX model:

    trtexec --onnx=onnx/light_vfi_base_544x960.onnx ^
            --saveEngine=trt/light_vfi_base_544x960_fp16.engine ^
            --fp16 ^
            --workspace=4096

This produces a `.engine` file optimized for FP16 inference.

### 9.2 TensorRT Inference

Run interpolation using the TensorRT engine:

    python infer_trt_light_vfi.py ^
        --engine trt/light_vfi_base_544x960_fp16.engine ^
        --input videos_train/input.mp4 ^
        --output outputs/output_interp_trt.mp4 ^
        --internal_h 544 ^
        --internal_w 960

Compared to pure PyTorch, TensorRT can significantly increase throughput and reduce latency, making near real-time interpolation more feasible.

---

## 10. Model Architecture (UNet2D)

The core model `LightVFIUNet2D` is a lightweight UNet-style network:

- 3-level encoder  
- Bottleneck block  
- 3-level decoder with skip connections  
- Input: 4 RGB frames stacked along the channel dimension → 12-channel input  
- Output: 1 RGB frame (the interpolated middle frame)  

Design goals:

- Simple and stable  
- Efficient on modern GPUs  
- Easy to extend with LoRA and export to ONNX  

---

## 11. LoRA Integration

The project integrates LoRA as follows:

- Standard Conv2D layers in selected parts of the network are replaced with LoRA-augmented Conv2D modules.  
- During LoRA training:
  - Base model parameters are frozen (`requires_grad = False`)  
  - Only LoRA parameters are updated  
- At inference time:
  - Base checkpoint is loaded  
  - LoRA checkpoint is optionally loaded and merged on top  

This enables:

- Multi-scene adaptation (different LoRA modules per game/scene/domain)  
- Lower update cost compared to full fine-tuning  
- Experimental comparison between full fine-tuning vs. LoRA-based fine-tuning  

---

## 12. Applications

Potential applications of Light_VFI include:

- Video frame interpolation (e.g., 24/30/60 FPS → higher frame-rate)  
- Smoothing animation or game footage  
- Offline enhancement of recorded gameplay videos  
- Prototype pipeline for DLSS-like frame generation research  
- Teaching and research on neural rendering, video SR/VFI, and LoRA techniques  

---

## 13. Citation

If you use this project in academic work or presentations, you may cite it as:

Light_VFI: Lightweight Video Frame Interpolation System (2025),  
Author: fallashOuO.

---

## 14. Future Work

Possible extensions and research directions:

- Optical-flow-guided refinement for motion-aware interpolation  
- Multi-stage or multi-frame interpolation (×4, ×8, etc.)  
- Hybrid CNN + Transformer architecture  
- TensorRT INT8 quantization for further speed-up  
- Integration with live screen capture / OBS streaming for real-time game or camera interpolation  

---

## 15. License

You may attach any license you prefer (e.g., MIT) depending on how you plan to distribute and reuse the code and models.
