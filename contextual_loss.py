
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextualLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0):
        super(ContextualLoss, self).__init__()
        self.sigma = sigma
        self.b = b

    def forward(self, x, y):
        assert x.shape == y.shape, "Inputs must have the same shape"
        B, C, H, W = x.shape

        x_flat = x.view(B, C, -1)
        y_flat = y.view(B, C, -1)

        x_flat = F.normalize(x_flat, dim=1)
        y_flat = F.normalize(y_flat, dim=1)

        dist = torch.cdist(x_flat.transpose(1, 2), y_flat.transpose(1, 2), p=2)  # [B, HW, HW]
        d_min = dist.min(dim=-1, keepdim=True)[0]
        d_scaled = dist / (d_min + 1e-5)

        w = torch.exp((self.b - d_scaled) / self.sigma)
        cx = w / torch.sum(w, dim=-1, keepdim=True)

        cx_loss = -torch.log(torch.mean(torch.max(cx, dim=-1)[0] + 1e-5))
        return cx_loss
