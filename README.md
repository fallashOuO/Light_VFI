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

