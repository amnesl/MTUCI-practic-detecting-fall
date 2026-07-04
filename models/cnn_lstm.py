"""
Модель CNN + LSTM для детекции падения
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTM(nn.Module):
    """CNN + LSTM для классификации видеопоследовательностей"""
    
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2):
        super(CNNLSTM, self).__init__()
        
        # ResNet50 как экстрактор признаков (заморожен)
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # убираем классификационную голову
        # Замораживаем веса CNN
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # LSTM для временной обработки
        self.lstm = nn.LSTM(
            input_size=2048,  # выход ResNet50
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Классификационная голова
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, T, C, H, W)
        batch, T, C, H, W = x.shape
        x = x.view(batch * T, C, H, W)
        
        # Извлечение признаков CNN (без градиентов)
        with torch.no_grad():
            features = self.cnn(x)  # (batch * T, 2048)
        
        features = features.view(batch, T, -1)  # (batch, T, 2048)
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # (batch, T, hidden*2)
        
        # Берем последний временной шаг
        output = self.fc(lstm_out[:, -1, :])
        return output