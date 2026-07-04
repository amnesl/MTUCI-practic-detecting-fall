"""
SlowFast модель для детекции падения
"""

import torch
import torch.nn as nn
import torch.hub


def get_slowfast_model(num_classes=2):
    """Загрузка и адаптация SlowFast для 2 классов"""
    
    # Загрузка модели через PyTorch Hub
    model = torch.hub.load(
        'facebookresearch/pytorchvideo',
        'slowfast_r50',
        pretrained=True
    )
    
    # Адаптация под 2 класса
    model.blocks[-1].proj = nn.Linear(2048, num_classes)
    
    return model