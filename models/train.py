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


def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-4, device='cuda', save_dir='runs/model'):
    """
    Обучение модели с сохранением лучших весов
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0.0
    patience = 5  # сколько эпох ждать улучшения
    patience_counter = 0
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
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
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
        
        # Сохраняем историю
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)
        history['val_precision'].append(prec)
        history['val_recall'].append(rec)
        history['val_f1'].append(f1)
        
        # Сохраняем лучшую модель
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            patience_counter = 0
            print(f"  ✅ Сохранена новая лучшая модель (F1={f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️ Early stopping на эпохе {epoch+1}")
                break
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"F1: {f1:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    # Сохраняем историю обучения
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    
    return history, best_f1


def evaluate_model(model, test_loader, device='cuda', save_dir='runs/model'):
    """
    Оценка модели на тестовой выборке
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Загружаем лучшие веса
    model.load_state_dict(torch.load(Path(save_dir) / 'best_model.pth'))
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
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
    
    # 🔧 ПРЕОБРАЗУЕМ ВСЕ В PYTHON-ТИПЫ ДЛЯ JSON 🔧
    results = {
        'confusion_matrix': cm.tolist(),  # Преобразуем numpy array в list
        'TP': int(TP),     # int64 → int
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'accuracy': float(acc),   # float64 → float
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    
    # Сохраняем результаты
    with open(Path(save_dir) / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n📊 Результаты на тестовой выборке:")
    print(f"  TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    return results