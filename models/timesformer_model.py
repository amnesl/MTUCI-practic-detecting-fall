"""
TimeSformer модель для детекции падения
"""

import torch.nn as nn
from transformers import TimesformerForVideoClassification


def get_timesformer_model(num_classes=2):
    """Загрузка и адаптация TimeSformer для 2 классов"""
    
    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        num_labels=num_classes
    )
    
    return model