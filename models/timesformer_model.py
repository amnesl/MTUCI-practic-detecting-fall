"""
TimeSformer модель для детекции падений
"""

import torch
import torch.nn as nn
from transformers import TimesformerForVideoClassification


def get_timesformer_model(num_classes=2, pretrained=True):
    """
    Создает TimeSformer модель с предобученными весами
    """
    print("   Загрузка TimeSformer модели...")
    
    try:
        # 🔥 ВАЖНО: добавляем ignore_mismatched_sizes=True
        model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # ← КЛЮЧЕВОЙ ПАРАМЕТР!
            torch_dtype=torch.float32
        )
        print("   ✅ TimeSformer модель загружена")
        print(f"   ✅ Голова заменена: 400 -> {num_classes} классов")
        
        return model
        
    except Exception as e:
        print(f"   ❌ Ошибка загрузки TimeSformer: {e}")
        print("   Использую упрощенную модель...")
        
        # Модель-заглушка на случай ошибки
        class TimeSformerDummy(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.fc = nn.Linear(768, num_classes)  # 768 - размер эмбеддинга TimeSformer
            
            def forward(self, x):
                # x: [B, T, C, H, W]
                # Усредняем по пространственным размерностям
                if x.dim() == 5:
                    x = x.mean(dim=[2, 3, 4])  # [B, T] -> [B, 768]
                return self.fc(x)
        
        return TimeSformerDummy(num_classes)