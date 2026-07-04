"""
3D ResNet модель для детекции падения
"""

import torch
import torch.nn as nn
import torch.hub


class ResNet3DWrapper(nn.Module):
    """Обертка для 3D ResNet, которая меняет порядок размерностей"""
    
    def __init__(self, num_classes=2):
        super(ResNet3DWrapper, self).__init__()
        self.model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'slow_r50',
            pretrained=True
        )
        # Адаптация под 2 класса
        self.model.blocks[-1].proj = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # x: (batch, T, C, H, W) → (batch, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)


def get_resnet3d_model(num_classes=2):
    """Загрузка и адаптация 3D ResNet для 2 классов с оберткой"""
    return ResNet3DWrapper(num_classes=2)