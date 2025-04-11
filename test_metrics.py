# test_metrics.py

import torch
from metrics import compute_psnr, compute_ssim

# 建立模擬的 HR 與 SR 圖像
gt = torch.rand(1, 1, 64, 64)  # HR ground truth
pred = gt + 0.05 * torch.randn_like(gt)  # SR with noise

# Clip 預測值到 [0, 1]
pred = torch.clamp(pred, 0.0, 1.0)

# 計算 PSNR 和 SSIM
psnr = compute_psnr(gt, pred)
ssim = compute_ssim(gt, pred)

print(f"[TEST] PSNR: {psnr:.2f} dB")
print(f"[TEST] SSIM: {ssim:.4f}")
