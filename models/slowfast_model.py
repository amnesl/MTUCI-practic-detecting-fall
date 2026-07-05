"""
SlowFast модель для детекции падений (кастомная версия)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlowFastCustom(nn.Module):
    """
    Упрощенная реализация SlowFast для детекции падений
    """
    def __init__(self, num_classes=2, slow_depth=4, fast_depth=8):
        super().__init__()
        
        # Slow pathway (низкая частота кадров)
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Fast pathway (высокая частота кадров) - в 2 раза больше кадров
        self.fast_conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.fast_bn1 = nn.BatchNorm3d(64)
        self.fast_pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Простые блоки для обработки
        self.slow_block = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((slow_depth, 7, 7))
        )
        
        self.fast_block = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((fast_depth, 7, 7))
        )
        
        # Fusion: объединение slow и fast путей
        self.fusion = nn.Conv3d(128 * 2, 256, kernel_size=1)
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: [B, T, C, H, W] из датасета
        # Преобразуем в [B, C, T, H, W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        # Slow путь: берем каждый 4-й кадр
        slow_indices = torch.arange(0, T, 4, device=x.device)
        if len(slow_indices) == 0:
            slow_indices = torch.tensor([0], device=x.device)
        slow_input = x[:, :, slow_indices, :, :]  # [B, C, T/4, H, W]
        
        # Fast путь: берем каждый 2-й кадр
        fast_indices = torch.arange(0, T, 2, device=x.device)
        if len(fast_indices) == 0:
            fast_indices = torch.tensor([0], device=x.device)
        fast_input = x[:, :, fast_indices, :, :]  # [B, C, T/2, H, W]
        
        # Slow pathway
        slow_out = self.slow_conv1(slow_input)
        slow_out = self.slow_bn1(slow_out)
        slow_out = F.relu(slow_out)
        slow_out = self.slow_pool1(slow_out)
        slow_out = self.slow_block(slow_out)
        
        # Fast pathway
        fast_out = self.fast_conv1(fast_input)
        fast_out = self.fast_bn1(fast_out)
        fast_out = F.relu(fast_out)
        fast_out = self.fast_pool1(fast_out)
        fast_out = self.fast_block(fast_out)
        
        # Fusion: изменяем размерности для конкатенации
        if slow_out.shape[2] != fast_out.shape[2]:
            fast_out = F.interpolate(fast_out, size=(slow_out.shape[2], 7, 7), mode='trilinear', align_corners=False)
        
        # Объединяем по каналам
        combined = torch.cat([slow_out, fast_out], dim=1)  # [B, 256, T, 7, 7]
        
        # Fusion
        fused = self.fusion(combined)
        
        # Классификация
        output = self.classifier(fused)
        
        return output


def get_slowfast_model(num_classes=2, pretrained=True):
    """
    Создает кастомную SlowFast модель (не использует PyTorchVideo)
    """
    print("   Создание кастомной SlowFast модели...")
    model = SlowFastCustom(num_classes=num_classes)
    
    if pretrained:
        print("   ⚠️ Предобученные веса не загружены (используются случайные)")
        print("   Рекомендуется обучить модель на вашем датасете")
    
    print(f"   ✅ SlowFast модель создана (классов: {num_classes})")
    return model