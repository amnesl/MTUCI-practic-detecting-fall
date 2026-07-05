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
    
    def __init__(self, data_dir, transform=None, frames_per_clip=16):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.frames_per_clip = frames_per_clip  # ← ДОБАВЛЕНО
        self.samples = []
        self.labels = []
        
        # Загрузка файлов: класс 0 = adl, класс 1 = fall
        for label, class_name in enumerate(["adl", "fall"]):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for file_path in class_dir.glob("*.npy"):
                    self.samples.append(file_path)
                    self.labels.append(label)
        
        print(f"✅ Загружено {len(self.samples)} примеров из {data_dir}")
        print(f"   adl (0): {self.labels.count(0)} примеров")
        print(f"   fall (1): {self.labels.count(1)} примеров")
        print(f"   Кадров в клипе: {self.frames_per_clip}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])  # (16, 3, 224, 224)
        
        # 🔥 НОВОЕ: обрезаем или дополняем до нужного числа кадров
        current_frames = data.shape[0]
        if current_frames > self.frames_per_clip:
            # Если кадров больше - берем первые N
            data = data[:self.frames_per_clip]
        elif current_frames < self.frames_per_clip:
            # Если кадров меньше - дополняем повторением последнего кадра
            pad = self.frames_per_clip - current_frames
            last_frame = data[-1:].repeat(pad, axis=0)
            data = np.concatenate([data, last_frame], axis=0)
        
        label = self.labels[idx]
        
        # Конвертируем в torch tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return data, label


def get_dataloaders(data_path, batch_size=4, num_workers=0, frames_per_clip=16):
    """
    Создает DataLoader для train, val, test
    
    Args:
        data_path: путь к папке с данными
        batch_size: размер батча
        num_workers: количество потоков загрузки
        frames_per_clip: количество кадров в клипе (для TimeSformer можно 8)
    """
    
    train_dataset = FallDetectionDataset(
        data_path / "train", 
        frames_per_clip=frames_per_clip
    )
    val_dataset = FallDetectionDataset(
        data_path / "val", 
        frames_per_clip=frames_per_clip
    )
    test_dataset = FallDetectionDataset(
        data_path / "test", 
        frames_per_clip=frames_per_clip
    )
    
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
        shuffle=True,  # 🔥 Перемешиваем test
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader