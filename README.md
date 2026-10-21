# Light_VFI

Light_VFI 是一個使用 PyTorch 實作的輕量化影片補幀研究專案。

模型會使用目標畫面前後共四張影格：

```text
[i - 2, i - 1, i + 1, i + 2]
```

預測中間缺少的目標影格：

```text
i
```

專案目前支援：

- 基礎模型從頭訓練
- Full Fine-tuning
- BitFit 微調
- LoRA 微調
- PSNR 與 SSIM 評估
- 影片抽幀
- GPU/CUDA 環境測試

## 專案結構

```text
Light_VFI/
├─ experiments/
│  ├─ __init__.py
│  ├─ _train_common.py
│  ├─ train_fullft.py
│  ├─ train_bitfit.py
│  ├─ train_lora.py
│  └─ evaluate.py
├─ src/
│  ├─ __init__.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ dataset.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ light_vfi.py
│  ├─ peft/
│  │  ├─ __init__.py
│  │  └─ lora.py
│  ├─ losses.py
│  └─ utils.py
├─ tools/
│  └─ extract_frames.py
├─ .gitignore
├─ requirements.txt
├─ testGPU.py
└─ README.md
```

## 環境需求

建議使用：

- Windows 11 或 Linux
- Python 3.10～3.11
- NVIDIA GPU
- 支援 CUDA 的 PyTorch

## 建立虛擬環境

在專案根目錄執行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 禁止執行啟用腳本，可以先執行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

安裝相依套件：

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 測試 GPU

```powershell
python .\testGPU.py
```

成功時應看到：

```text
CUDA matrix multiplication: PASSED
GPU environment test completed successfully.
```

## 準備資料集

每一部影片必須放在獨立資料夾中，而且每個資料夾至少需要五張連續影格。

範例：

```text
data/
├─ train/
│  ├─ video_001/
│  │  ├─ frame_000000.png
│  │  ├─ frame_000001.png
│  │  ├─ frame_000002.png
│  │  ├─ frame_000003.png
│  │  └─ frame_000004.png
│  └─ video_002/
│     ├─ frame_000000.png
│     ├─ frame_000001.png
│     ├─ frame_000002.png
│     ├─ frame_000003.png
│     └─ frame_000004.png
└─ test/
   └─ video_101/
      ├─ frame_000000.png
      ├─ frame_000001.png
      ├─ frame_000002.png
      ├─ frame_000003.png
      └─ frame_000004.png
```

訓練資料至少需要兩個有效的影片資料夾，因為程式會依照影片資料夾切分訓練集與驗證集，避免同一部影片的影格同時出現在兩邊。

支援的圖片格式：

```text
.png
.jpg
.jpeg
.bmp
.webp
```

影格檔名應使用可正確排序的編號，例如：

```text
frame_000001.png
frame_000002.png
frame_000003.png
```

不要使用：

```text
frame_1.png
frame_10.png
frame_2.png
```

否則字串排序可能破壞影格順序。

## 從影片抽取影格

保留所有影格：

```powershell
python .\tools\extract_frames.py `
  --video ".\videos\input.mp4" `
  --output ".\data\train\video_001" `
  --step 1 `
  --ext png
```

每兩張影格保留一張：

```powershell
python .\tools\extract_frames.py `
  --video ".\videos\input.mp4" `
  --output ".\data\train\video_001" `
  --step 2 `
  --ext png
```

`--step 2` 可以用於把 60 FPS 影片取樣成約 30 FPS 的影格序列。

## 訓練基礎模型

基礎模型使用 `train_base.py` 從頭訓練，不需要提供 `--base_ckpt`：

```powershell
python -m experiments.train_fullft `
  --data_root ".\data\train" `
  --save_dir ".\checkpoints\base" `
  --epochs 50 `
  --batch_size 8 `
  --lr 1e-4 `
  --resize 256 `
  --device cuda `
  --num_workers 4 `
  --seed 42
```

主要輸出：

```text
checkpoints/base/
├─ config.json
├─ epoch1.pth
├─ epoch2.pth
├─ ...
├─ best.pth
└─ final.pth
```

通常後續微調應使用：

```text
checkpoints/base/best.pth
```

## Full Fine-tuning

Full Fine-tuning 會載入基礎模型，並更新模型的所有參數：

```powershell
python -m experiments.train_fullft `
  --data_root ".\data\train" `
  --base_ckpt ".\checkpoints\base\best.pth" `
  --save_dir ".\checkpoints\fullft\domain1" `
  --epochs 5 `
  --batch_size 8 `
  --lr 1e-4 `
  --resize 256 `
  --device cuda `
  --num_workers 4 `
  --seed 42
```

## BitFit 微調

BitFit 只訓練模型中的 bias 參數：

```powershell
python -m experiments.train_bitfit `
  --data_root ".\data\train" `
  --base_ckpt ".\checkpoints\base\best.pth" `
  --save_dir ".\checkpoints\bitfit\domain1" `
  --epochs 5 `
  --batch_size 8 `
  --lr 1e-4 `
  --resize 256 `
  --device cuda `
  --num_workers 4 `
  --seed 42
```

BitFit checkpoint 只保存接受訓練的參數，檔案通常會比完整模型小。

## LoRA 微調

LoRA 會凍結原始模型，只訓練注入的低秩參數：

```powershell
python -m experiments.train_lora `
  --data_root ".\data\train" `
  --base_ckpt ".\checkpoints\base\best.pth" `
  --save_dir ".\checkpoints\lora\domain1" `
  --epochs 5 `
  --batch_size 8 `
  --lr 1e-4 `
  --resize 256 `
  --device cuda `
  --num_workers 4 `
  --seed 42 `
  --r 8 `
  --alpha 16 `
  --dropout 0.0
```

LoRA checkpoint 只保存 LoRA 參數。評估時必須同時提供：

- 原始基礎模型 checkpoint
- LoRA checkpoint
- 與訓練相同的 `r`、`alpha` 和 `dropout`

## 評估基礎模型

```powershell
python -m experiments.evaluate `
  --data_root ".\data\test" `
  --base_ckpt ".\checkpoints\base\best.pth" `
  --method base `
  --batch_size 8 `
  --resize 256 `
  --device cuda
```

## 評估 Full Fine-tuning

```powershell
python -m experiments.evaluate `
  --data_root ".\data\test" `
  --base_ckpt ".\checkpoints\base\best.pth" `
  --method fullft `
  --ckpt ".\checkpoints\fullft\domain1\best.pth" `
  --batch_size 8 `
  --resize 256 `
  --device cuda
```

## 評估 BitFit

```powershell
python -m experiments.evaluate `
  --data_root ".\data\test" `
  --base_ckpt ".\checkpoints\base\best.pth" `
  --method bitfit `
  --ckpt ".\checkpoints\bitfit\domain1\best.pth" `
  --batch_size 8 `
  --resize 256 `
  --device cuda
```

## 評估 LoRA

```powershell
python -m experiments.evaluate `
  --data_root ".\data\test" `
  --base_ckpt ".\checkpoints\base\best.pth" `
  --method lora `
  --ckpt ".\checkpoints\lora\domain1\best.pth" `
  --batch_size 8 `
  --resize 256 `
  --device cuda `
  --r 8 `
  --alpha 16 `
  --dropout 0.0
```

評估結果預設會追加到：

```text
results/metrics/metrics.csv
```

包含：

- 方法名稱
- 資料集路徑
- checkpoint 路徑
- checkpoint 大小
- PSNR
- SSIM

## 訓練紀錄

每次訓練完成後，結果會追加到：

```text
results/logs/train_log.csv
```

紀錄內容包括：

- 訓練方法
- epoch 數量
- batch size
- learning rate
- 可訓練參數量
- 總參數量
- 最佳驗證 loss
- 最佳 epoch
- 訓練時間
- checkpoint 大小

## 常見問題

### `No module named 'src'`

請確認目前位於專案根目錄，並使用模組方式執行：

```powershell
python -m experiments.train_lora
```

不要在 `experiments` 資料夾內直接執行程式。

### `CUDA not available`

先執行：

```powershell
python .\testGPU.py
```

並確認安裝的是支援 CUDA 的 PyTorch，而不是 CPU-only 版本。

### `No valid samples`

每個影片資料夾至少需要五張可讀取且順序正確的影格。

### CUDA 記憶體不足

可以依序嘗試：

1. 降低 `--batch_size`
2. 降低 `--resize`
3. 關閉其他占用 GPU 的程式

例如：

```powershell
--batch_size 4 --resize 192
```

## 方法比較

| 方法 | 訓練參數 | Checkpoint 大小 | 適用情況 |
|---|---:|---:|---|
| Base | 全部參數 | 大 | 建立基礎模型 |
| Full Fine-tuning | 全部參數 | 大 | 追求完整領域適配 |
| BitFit | Bias 參數 | 小 | 低成本微調 |
| LoRA | 低秩參數 | 小 | 多領域快速適配 |

## 注意事項

- Vimeo90K Triplet 只有三張影格，不符合目前四張輸入影格的模型設計，因此目前不使用。
- `data`、`videos`、`checkpoints` 和 `results` 中的大型檔案不應上傳至 GitHub。
- 執行訓練與評估指令時，工作目錄必須位於專案根目錄。
- LoRA 評估參數必須與 LoRA 訓練時完全一致。
- 建議固定使用 `--seed 42`，方便重現實驗結果。