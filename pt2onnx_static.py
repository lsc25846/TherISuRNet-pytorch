import os
import random
import torch
import numpy as np
from PIL import Image
from model import TherISuRNet
from torchvision import transforms
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import onnx

# === 1. 匯出浮點 ONNX ===
model = TherISuRNet(scale=4)
ckpt = torch.load("checkpoints/latest.pth", map_location="cpu")["model"]
model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
model.eval()

dummy_input = torch.randn(1, 1, 128, 160)
onnx_path = "therisurnet_fp32.onnx"
torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], opset_version=13)
print(f"✅ Exported FP32 ONNX: {onnx_path}")

# === 2. 建立靜態量化代表資料集 ===
class LRImageReader(CalibrationDataReader):
    def __init__(self, image_dir):
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 160)),
            transforms.ToTensor()
        ])
        all_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.images = random.sample(all_images, min(100, len(all_images)))
        self.image_dir = image_dir
        self.index = 0

    def get_next(self):
        if self.index >= len(self.images):
            return None
        img_path = os.path.join(self.image_dir, self.images[self.index])
        img = Image.open(img_path).convert("L")
        tensor = self.transform(img).unsqueeze(0).numpy().astype(np.float32)
        self.index += 1
        return {"input": tensor}

reader = LRImageReader("train_lr")

# === 3. 執行靜態量化 ===
quantized_model_path = "therisurnet_int8.onnx"
quantize_static(
    model_input=onnx_path,
    model_output=quantized_model_path,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print(f"✅ INT8 ONNX saved to: {quantized_model_path}")
