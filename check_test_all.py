"""
Проверка ВСЕХ батчей в test
"""

from pathlib import Path
from models.dataset import get_dataloaders

data_path = Path("./prepared_data")

# Загружаем данные с перемешиванием
train_loader, val_loader, test_loader = get_dataloaders(
    data_path, 
    batch_size=16, 
    num_workers=0
)

print("\n" + "="*50)
print("ПРОВЕРКА ВСЕХ БАТЧЕЙ (test)")
print("="*50)

total_adl = 0
total_fall = 0
batches_with_fall = 0
batches_without_fall = 0

for i, (batch, labels) in enumerate(test_loader):
    adl_count = (labels == 0).sum().item()
    fall_count = (labels == 1).sum().item()
    
    total_adl += adl_count
    total_fall += fall_count
    
    if fall_count > 0:
        batches_with_fall += 1
    else:
        batches_without_fall += 1
    
    if i < 5 or fall_count > 0:  # Показываем первые 5 и все с fall
        print(f"Батч {i}: adl={adl_count}, fall={fall_count}")

print("\n" + "="*50)
print(f"ИТОГО:")
print(f"  adl: {total_adl}")
print(f"  fall: {total_fall}")
print(f"  Батчей с fall: {batches_with_fall}")
print(f"  Батчей без fall: {batches_without_fall}")

if batches_with_fall == 0:
    print("\n❌ ОШИБКА: В test нет ни одного батча с классом fall!")
    print("   Возможно, датасет собран неправильно или папка test/fall пуста")