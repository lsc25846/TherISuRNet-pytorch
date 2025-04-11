
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TherISuRNet
from metrics import compute_psnr, compute_ssim
from contextual_loss import ContextualLoss
from thermal_dataset import ThermalImageDataset
from torchvision.utils import make_grid, save_image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "checkpoints"
SAMPLE_DIR = "sample"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for lr, hr in dataloader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        sr = torch.nn.functional.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
        loss = criterion(sr, hr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_psnr, total_ssim = 0, 0
    for lr, hr in dataloader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        sr = torch.nn.functional.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
        total_psnr += compute_psnr(hr, sr)
        total_ssim += compute_ssim(hr, sr)
    return total_psnr / len(dataloader), total_ssim / len(dataloader)

@torch.no_grad()
def save_sample_images(model, dataset, epoch):
    model.eval()
    lr, hr = dataset[0]  # demo sample
    lr, hr = lr.unsqueeze(0).to(device), hr.unsqueeze(0).to(device)
    sr = model(lr)
    sr = torch.nn.functional.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
    lr_resized = torch.nn.functional.interpolate(lr, size=hr.shape[2:], mode='bicubic', align_corners=False)
    grid = make_grid(torch.cat([lr_resized, sr, hr], dim=0), nrow=3, normalize=True)
    save_image(grid, os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}.png"))

def main(args):
    model = TherISuRNet(scale=args.scale)

    # ✅ 多 GPU 支援（DataParallel）
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs via nn.DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)

    
    train_dataset = ThermalImageDataset(os.path.join(args.data_dir, "train_lr"), os.path.join(args.data_dir, "train_hr"))
val_dataset = ThermalImageDataset(os.path.join(args.data_dir, "val_lr"), os.path.join(args.data_dir, "val_hr"))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    l1 = nn.L1Loss()
    cx = ContextualLoss()

    def combined_loss(sr, hr):
        sr_down = torch.nn.functional.interpolate(sr, size=(64, 64), mode='bilinear', align_corners=False)
        hr_down = torch.nn.functional.interpolate(hr, size=(64, 64), mode='bilinear', align_corners=False)
        return 10 * l1(sr, hr) + 0.1 * cx(sr_down, hr_down)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 1

    if args.weights and os.path.exists(args.weights):
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss = train_one_epoch(model, train_loader, combined_loss, optimizer)
        psnr, ssim = evaluate(model, val_loader)
        print(f"[Epoch {epoch}] Loss: {loss:.4f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
        if epoch % 5 == 0:
            save_sample_images(model, val_dataset, epoch)

        torch.save({
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, os.path.join(SAVE_DIR, "latest.pth"))
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--weights", type=str, default=None, help="Resume weights from .pth checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--data_dir", type=str, default=".", help="Root directory containing train_lr, train_hr, val_lr, val_hr")
    args = parser.parse_args()
    main(args)
