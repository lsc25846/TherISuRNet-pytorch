
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ThermalImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.jpg')])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.jpg')])
        assert len(self.lr_files) == len(self.hr_files), "LR and HR count mismatch"
        self.transform = transform or T.Compose([
            T.ToTensor(),  # 0~1
        ])

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        
        lr = Image.open(lr_path).convert('L')  # 灰階
        hr = Image.open(hr_path).convert('L')

        lr = self.transform(lr)
        hr = self.transform(hr)
        return lr, hr
