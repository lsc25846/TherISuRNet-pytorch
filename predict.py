
import os
import torch
from model import TherISuRNet
from torchvision import transforms
from PIL import Image
import argparse

def load_image(path):
    img = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # shape: [1, 1, H, W]

def save_image_tensor(tensor, path):
    tensor = tensor.squeeze(0).clamp(0, 1)
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor.cpu())
    img.save(path)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TherISuRNet(scale=args.scale).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # 預測每張 LR 圖片
    lr_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".jpg")])
    for fname in lr_files:
        lr_path = os.path.join(args.input_dir, fname)
        lr = load_image(lr_path).to(device)

        with torch.no_grad():
            sr = model(lr)
            sr = torch.nn.functional.interpolate(sr, scale_factor=args.scale, mode='bicubic', align_corners=False)

        out_path = os.path.join(args.output_dir, f"SR_{fname}")
        save_image_tensor(sr, out_path)
        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to LR .jpg images")
    parser.add_argument("--output_dir", type=str, default="results", help="Where to save SR images")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth model weights")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor used in training")
    args = parser.parse_args()
    main(args)
