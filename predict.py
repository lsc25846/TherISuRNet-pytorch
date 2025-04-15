import os
import argparse
import torch
import torch.nn.functional as F
from model import TherISuRNet
from torchvision.utils import save_image
from collections import OrderedDict
from PIL import Image
import numpy as np
from torchvision import transforms
from metrics import compute_psnr, compute_ssim

def load_image(path):
    img = Image.open(path).convert("L")
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # [1, 1, H, W]

def save_image_tensor(tensor, path):
    tensor = tensor.squeeze(0).clamp(0, 1)
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor.cpu())
    img.save(path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict_onnx(args):
    import onnxruntime as ort
    #ort_session = ort.InferenceSession(args.weights)
    providers = ['CPUExecutionProvider']  # 如果你有安裝 OpenVINO / QNN 可放在前面
    ort_session = ort.InferenceSession(args.weights, providers=providers)


    lr_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".jpg")])
    os.makedirs(args.output_dir, exist_ok=True)

    total_psnr, total_ssim = 0, 0
    for fname in lr_files:
        lr = load_image(os.path.join(args.input_dir, fname))
        lr_np = to_numpy(lr)

        ort_inputs = {ort_session.get_inputs()[0].name: lr_np}
        ort_outs = ort_session.run(None, ort_inputs)
        sr = torch.tensor(ort_outs[0])
        #sr = F.interpolate(sr, scale_factor=args.scale, mode='bicubic', align_corners=False)
        if sr.shape[-2:] == lr.shape[-2:]:  # 只在輸出圖大小等於輸入圖時放大
            sr = F.interpolate(sr, scale_factor=args.scale, mode='bicubic', align_corners=False)


        if args.hr_dir:
            hr_path = os.path.join(args.hr_dir, fname)
            if os.path.exists(hr_path):
                hr = load_image(hr_path)
                if sr.shape != hr.shape:
                    sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
                total_psnr += compute_psnr(hr, sr)
                total_ssim += compute_ssim(hr, sr)

        out_path = os.path.join(args.output_dir, f"SR_{fname}")
        save_image_tensor(sr, out_path)
        print(f"✅ Saved: {out_path}")

    if args.hr_dir:
        n = len(lr_files)
        print(f"📊 ONNX INT8 PSNR: {total_psnr/n:.2f} | SSIM: {total_ssim/n:.4f}")

def predict_pth(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TherISuRNet(scale=args.scale).to(device)

    print(f"🔄 Loading weights from: {args.weights}")
    state_dict = torch.load(args.weights, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict)
    model.eval()

    lr_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".jpg")])
    os.makedirs(args.output_dir, exist_ok=True)

    total_psnr, total_ssim = 0, 0
    for fname in lr_files:
        lr = load_image(os.path.join(args.input_dir, fname)).to(device)

        with torch.no_grad():
            sr = model(lr)
            #sr = F.interpolate(sr, scale_factor=args.scale, mode='bicubic', align_corners=False)
            if sr.shape[-2:] == lr.shape[-2:]:  # 只在輸出圖大小等於輸入圖時放大
                sr = F.interpolate(sr, scale_factor=args.scale, mode='bicubic', align_corners=False)


        if args.hr_dir:
            hr_path = os.path.join(args.hr_dir, fname)
            if os.path.exists(hr_path):
                hr = load_image(hr_path).to(device)
                if sr.shape != hr.shape:
                    sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
                total_psnr += compute_psnr(hr, sr)
                total_ssim += compute_ssim(hr, sr)

        out_path = os.path.join(args.output_dir, f"SR_{fname}")
        save_image_tensor(sr, out_path)
        print(f"✅ Saved: {out_path}")

    if args.hr_dir:
        n = len(lr_files)
        print(f"📊 Float32 PSNR: {total_psnr/n:.2f} | SSIM: {total_ssim/n:.4f}")

def main(args):
    if args.use_onnx:
        predict_onnx(args)
    else:
        predict_pth(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with LR .jpg images")
    parser.add_argument("--output_dir", type=str, default="results", help="Where to save SR images")
    parser.add_argument("--hr_dir", type=str, help="Optional: GT HR folder for PSNR/SSIM")
    parser.add_argument("--weights", type=str, required=True, help=".pth or .onnx model path")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor (used for bicubic resize)")
    parser.add_argument("--use_onnx", action="store_true", help="Use ONNX INT8 inference")
    args = parser.parse_args()

    main(args)
