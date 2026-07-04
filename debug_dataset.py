from pathlib import Path
from models.dataset import FallDetectionDataset

dataset = FallDetectionDataset(Path("./prepared_data/train"))

print(f"Всего примеров: {len(dataset)}")

# Проверка первых 10 меток
for i in range(10):
    _, label = dataset[i]
    print(f"Пример {i}: метка = {label.item()}")

# Проверка распределения меток
labels = [dataset[i][1].item() for i in range(len(dataset))]
print(f"\nКласс 0 (ADL): {labels.count(0)}")
print(f"Класс 1 (Fall): {labels.count(1)}")