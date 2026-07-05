"""
Проверка батчей из DataLoader
"""

from pathlib import Path
from models.dataset import get_dataloaders

data_path = Path("./prepared_data")

# Загружаем данные с batch_size=16 (как в обучении)
train_loader, val_loader, test_loader = get_dataloaders(
    data_path, 
    batch_size=16, 
    num_workers=0
)

print("\n" + "="*50)
print("ПРОВЕРКА БАТЧЕЙ (train)")
print("="*50)

for i, (batch, labels) in enumerate(train_loader):
    print(f"Батч {i}:")
    print(f"  Тензор: {batch.shape}")  # Ожидается [16, 16, 3, 224, 224]
    print(f"  Метки: {labels.tolist()}")
    print(f"  Класс 0 (adl): {(labels == 0).sum().item()} шт.")
    print(f"  Класс 1 (fall): {(labels == 1).sum().item()} шт.")
    print()
    
    if i == 3:  # Проверим первые 4 батча
        break

print("\n" + "="*50)
print("ПРОВЕРКА БАТЧЕЙ (test)")
print("="*50)

for i, (batch, labels) in enumerate(test_loader):
    print(f"Батч {i}:")
    print(f"  Метки: {labels.tolist()}")
    print(f"  Класс 0: {(labels == 0).sum().item()} шт.")
    print(f"  Класс 1: {(labels == 1).sum().item()} шт.")
    
    if i == 3:
        break