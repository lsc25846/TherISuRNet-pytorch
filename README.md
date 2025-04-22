# TherISuRNet: Thermal Image Super-Resolution Network

TherISuRNet 是一個用於 **熱影像超解析 (Thermal Image Super-Resolution)** 的深度學習模型，使用 PyTorch 實現。
本模型可將低解析度的熱影像重建為高解析度版本，適用於行車輔助、監控與低光場景下的物體辨識前處理。

---

## 🧠 專案特色

- 基於自定義卷積模型設計（見 `model.py`）
- 支援 **Contextual Loss (CX Loss)** 以提升感知品質
- 內建 `PSNR`, `SSIM` 評估指標
- 提供訓練與推論流程
- 支援模型轉換為 ONNX 格式

---

## 📦 安裝方式

```bash
git clone https://github.com/your_repo/Therisurnet_pytorch.git
cd Therisurnet_pytorch
pip install -r requirement.txt
```

建議使用 Python 3.8 以上 + CUDA 支援的 PyTorch 環境。

---

## 📁 專案結構

```
Therisurnet_pytorch/
├── train.py               # 訓練腳本
├── predict.py             # 推論腳本
├── model.py               # TherISuRNet 模型
├── thermal_dataset.py     # Dataset 定義（LR/HR 配對）
├── contextual_loss.py     # Contextual Loss 損失函數
├── PSNR_SSIM.py           # 評估指標實作
├── pt2onnx_static.py      # PyTorch → ONNX 轉換工具
├── checkpoints/           # 模型儲存
├── sample/                # 測試樣本
├── results/               # 推論輸出結果
├── train_hr/, train_lr/   # 訓練資料夾（HR/LR）
├── val_hr/, val_lr/       # 驗證資料夾（HR/LR）
└── requirement.txt        # 套件需求
```

---

## 🚀 使用方式

### 1. 訓練模型

確保以下資料夾有正確影像：
- `train_hr/`：高解析度訓練影像
- `train_lr/`：對應的低解析度影像

執行訓練指令：

```bash
python train.py
```

模型會儲存至 `checkpoints/` 資料夾。

---

### 2. 執行推論

將測試圖片放入 `sample/` 資料夾，執行：

```bash
python predict.py
```

輸出影像將儲存於 `results/` 資料夾。

---

### 3. 模型轉換為 ONNX

```bash
python pt2onnx_static.py
```

---

### 4. 評估指標

使用 PSNR 與 SSIM 指標評估模型：

```bash
python test_metrics.py
```

---

## 🧪 損失函數：Contextual Loss

本模型支援 [Contextual Loss](https://arxiv.org/abs/1803.02077) 增加影像感知品質。可於 `train.py` 中控制開關：

```python
use_cx_loss = True
```

---

## ✅ 相依套件

請參考 `requirement.txt`，主要依賴：

```
torch>=1.10
torchvision
scikit-image
opencv-python
numpy
tqdm
matplotlib
Pillow
onnx
onnxruntime
```

---

## 📬 聯絡與貢獻

歡迎提交 PR 或 Issue 討論與改進，如有合作研究需求也歡迎聯絡作者。

---

## 📸 範例輸出

| 原始 LR | TherISuRNet 輸出 HR |
|---------|--------------------|
| (sample image) | (super-resolved image) |

> *(請自行補上圖片展示)*

