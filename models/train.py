"""
Обучение модели для детекции падения
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def convert_numpy_to_python(obj):
    """Рекурсивно преобразует все numpy-типы в Python-типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, device='cuda', save_dir='runs/model'):
    """
    Обучение модели с сохранением лучших весов
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 🔥 ВЕСА КЛАССОВ для борьбы с дисбалансом
    class_weights = torch.tensor([1.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔥 ПРОВЕРКА: является ли модель из библиотеки transformers
    is_transformers_model = False
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        if model.config.model_type in ['timesformer', 'videomae']:
            is_transformers_model = True
            print(f"   🔍 Обнаружена модель {model.config.model_type}, использую pixel_values")
    
    best_f1 = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # ====== ОБУЧЕНИЕ ======
        model.train()
        train_loss = 0.0
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 🔥 Для моделей из transformers используем pixel_values
            if is_transformers_model:
                outputs = model(pixel_values=batch).logits
            else:
                outputs = model(batch)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 🔥 Градиентное клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ====== ВАЛИДАЦИЯ ======
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch, labels in val_loader:
                batch, labels = batch.to(device), labels.to(device)
                
                # 🔥 Для моделей из transformers используем pixel_values
                if is_transformers_model:
                    outputs = model(pixel_values=batch).logits
                else:
                    outputs = model(batch)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # ====== МЕТРИКИ ======
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        # Сохраняем историю (конвертируем в Python типы)
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(avg_val_loss))
        history['val_acc'].append(float(acc))
        history['val_precision'].append(float(prec))
        history['val_recall'].append(float(rec))
        history['val_f1'].append(float(f1))
        
        # Сохраняем лучшую модель
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"  ✅ Сохранена новая лучшая модель (F1={f1:.4f})")
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"F1: {f1:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    # Сохраняем историю обучения
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(convert_numpy_to_python(history), f, indent=4)
    
    return history, best_f1


def evaluate_model(model, test_loader, device='cuda', save_dir='runs/model'):
    """
    Оценка модели на тестовой выборке
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Загружаем лучшие веса
    model_path = Path(save_dir) / 'best_model.pth'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Загружены веса из {model_path}")
    else:
        print(f"⚠️ Внимание: файл {model_path} не найден, использую текущие веса")
    
    # 🔥 ПРОВЕРКА: является ли модель из библиотеки transformers
    is_transformers_model = False
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        if model.config.model_type in ['timesformer', 'videomae']:
            is_transformers_model = True
            print(f"   🔍 Обнаружена модель {model.config.model_type}, использую pixel_values")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            
            # 🔥 Для моделей из transformers используем pixel_values
            if is_transformers_model:
                outputs = model(pixel_values=batch).logits
            else:
                outputs = model(batch)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Метрики
    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    results = {
        'confusion_matrix': cm.tolist(),
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    
    # Сохраняем результаты
    with open(Path(save_dir) / 'test_results.json', 'w') as f:
        json.dump(convert_numpy_to_python(results), f, indent=4)
    
    print(f"\n📊 Результаты на тестовой выборке:")
    print(f"  TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    return results