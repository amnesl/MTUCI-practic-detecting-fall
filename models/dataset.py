"""
Загрузка данных для детекции падения
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random


class FallDetectionDataset(Dataset):
    """Датасет для детекции падения человека на видео"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Загрузка файлов: класс 0 = adl, класс 1 = fall
        for label, class_name in enumerate(["adl", "fall"]):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for file_path in class_dir.glob("*.npy"):
                    self.samples.append(file_path)
                    self.labels.append(label)
        
        print(f"Загружено {len(self.samples)} примеров из {data_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])  # (16, 3, 224, 224)
        label = self.labels[idx]
        
        # Конвертируем в torch tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label).long()
        
        return data, label


def get_dataloaders(data_path, batch_size=8, num_workers=8):
    """Создает DataLoader для train, val, test"""
    
    train_dataset = FallDetectionDataset(data_path / "train")
    val_dataset = FallDetectionDataset(data_path / "val")
    test_dataset = FallDetectionDataset(data_path / "test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader