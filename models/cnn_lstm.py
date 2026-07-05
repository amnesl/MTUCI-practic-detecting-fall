"""
CNN + LSTM модель для детекции падений
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super(CNNLSTM, self).__init__()
        
        # CNN backbone (замороженный)
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Замораживаем все слои
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # Размораживаем последние 2 слоя для тонкой настройки
        for param in list(self.cnn.layer4.parameters())[-10:]:
            param.requires_grad = True
            
        # Убираем классификатор ResNet
        self.cnn.fc = nn.Identity()
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,  # выход ResNet18
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Классификатор
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 128),  # *2 для bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, frames, channels, height, width]
        Returns:
            logits: [batch, num_classes]
        """
        batch, frames, C, H, W = x.shape
        
        # Изменяем размерность для CNN: [batch * frames, C, H, W]
        x = x.view(batch * frames, C, H, W)
        
        # Извлекаем признаки через CNN
        with torch.no_grad():
            features = self.cnn(x)  # [batch * frames, 512]
        
        # Возвращаем размерность: [batch, frames, 512]
        features = features.view(batch, frames, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # [batch, frames, lstm_hidden*2]
        
        # Берем последний кадр
        output = lstm_out[:, -1, :]  # [batch, lstm_hidden*2]
        
        # Классификация
        logits = self.fc(output)  # [batch, num_classes]
        
        return logits