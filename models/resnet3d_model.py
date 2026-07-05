"""
3D ResNet модель для детекции падений
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


def get_resnet3d_model(num_classes=2, pretrained=True):
    """
    Создает 3D ResNet модель с автоматическим преобразованием размерности
    """
    # Создаем базовую модель
    if pretrained:
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
    else:
        model = r3d_18(weights=None)
    
    # Заменяем классификатор
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    
    # 🔥 ОБЕРТКА: конвертирует [B, T, C, H, W] -> [B, C, T, H, W]
    class ResNet3DWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x):
            # x приходит в формате [B, T, C, H, W] (из dataset.py)
            # Нужно преобразовать в [B, C, T, H, W] для 3D ResNet
            if x.dim() == 5:
                # Проверяем: если второй размер (индекс 1) не равен 3 (каналы),
                # значит это скорее всего T (временная размерность)
                if x.shape[1] != 3 and x.shape[2] == 3:
                    x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
            
            return self.base_model(x)
    
    return ResNet3DWrapper(model)