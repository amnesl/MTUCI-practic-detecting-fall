"""
VideoMAE модель для детекции падения
"""

import torch.nn as nn
from transformers import VideoMAEForVideoClassification


def get_videomae_model(num_classes=2):
    """Загрузка и адаптация VideoMAE для 2 классов"""
    
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=num_classes
    )
    
    return model