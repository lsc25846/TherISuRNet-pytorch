# pytorch_metrics.py

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def tensor_to_np(tensor):
    """
    將 PyTorch tensor (1, 1, H, W) 轉換為 numpy 格式的 uint8 單通道圖片
    """
    tensor = tensor.squeeze().detach().cpu().numpy()
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    return tensor

def compute_psnr(gt_tensor, pred_tensor):
    """
    使用 skimage 計算 PSNR
    """
    gt_np = tensor_to_np(gt_tensor)
    pred_np = tensor_to_np(pred_tensor)
    return compare_psnr(gt_np, pred_np, data_range=255)

def compute_ssim(gt_tensor, pred_tensor):
    """
    使用 skimage 計算 SSIM
    """
    gt_np = tensor_to_np(gt_tensor)
    pred_np = tensor_to_np(pred_tensor)
    return compare_ssim(gt_np, pred_np, data_range=255, gaussian_weights=True)
