"""
Модель на основе ключевых точек + LSTM для детекции падения
"""

import torch
import torch.nn as nn


class PoseLSTM(nn.Module):
    """LSTM для классификации последовательностей ключевых точек"""
    
    def __init__(self, input_size=33*2, hidden_size=256, num_layers=3, num_classes=2):
        super(PoseLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, T, 66) — координаты 33 точек
        lstm_out, _ = self.lstm(x)  # (batch, T, hidden*2)
        output = self.fc(lstm_out[:, -1, :])
        return output


def extract_pose_sequence(video_path):
    """
    Извлечение ключевых точек из видео с помощью MediaPipe.
    Возвращает массив (T, 66) — 33 точки x 2 координаты.
    """
    import cv2
    import mediapipe as mp
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        
        if result.pose_landmarks:
            # Извлекаем координаты (x, y) для 33 точек
            landmarks = []
            for lm in result.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            keypoints.append(landmarks)
        else:
            # Если человек не обнаружен, добавляем нули
            keypoints.append([0.0] * 66)
    
    cap.release()
    return torch.tensor(keypoints).float()  # (T, 66)