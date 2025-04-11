
import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def tensor_to_np(tensor):
    tensor = tensor.squeeze().detach().cpu().numpy()
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    return tensor

def compute_psnr(gt_tensor, pred_tensor):
    gt_np = tensor_to_np(gt_tensor)
    pred_np = tensor_to_np(pred_tensor)
    return compare_psnr(gt_np, pred_np, data_range=255)

def compute_ssim(gt_tensor, pred_tensor):
    gt_np = tensor_to_np(gt_tensor)
    pred_np = tensor_to_np(pred_tensor)
    return compare_ssim(gt_np, pred_np, data_range=255, gaussian_weights=True)
