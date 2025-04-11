import torch
from model import TherISuRNet

def test_model(scale):
    print(f"[TEST] Testing TherISuRNet with scale ×{scale}")
    model = TherISuRNet(scale=scale)
    model.eval()

    # 假設原始低解析影像大小是 64x64
    lr = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        sr = model(lr)

    expected_size = 64 * scale
    assert sr.shape == (1, 1, expected_size, expected_size), \
        f"Expected output shape (1,1,{expected_size},{expected_size}), got {sr.shape}"

    print(f"[PASS] Output shape: {sr.shape}")

if __name__ == "__main__":
    for scale in [2, 3, 4]:
        test_model(scale)
