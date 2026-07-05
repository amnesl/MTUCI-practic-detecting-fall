"""
VideoMAE модель для детекции падений
"""

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification


def get_videomae_model(num_classes=2, pretrained=True):
    """
    Создает VideoMAE модель с предобученными весами
    """
    print("   Загрузка VideoMAE модели...")
    
    try:
        # 🔥 ВАЖНО: добавляем ignore_mismatched_sizes=True
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # ← КЛЮЧЕВОЙ ПАРАМЕТР!
            torch_dtype=torch.float32
        )
        print("   ✅ VideoMAE модель загружена")
        print(f"   ✅ Голова заменена: 400 -> {num_classes} классов")
        
        return model
        
    except Exception as e:
        print(f"   ❌ Ошибка загрузки VideoMAE: {e}")
        print("   Использую упрощенную модель...")
        
        # Модель-заглушка на случай ошибки
        class VideoMAEDummy(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.fc = nn.Linear(768, num_classes)
            
            def forward(self, x):
                if x.dim() == 5:
                    # Усредняем по пространственным размерностям
                    x = x.mean(dim=[2, 3, 4])
                return self.fc(x)
        
        return VideoMAEDummy(num_classes)