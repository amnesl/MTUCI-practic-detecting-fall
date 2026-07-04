"""
Запуск обучения всех моделей для детекции падения
"""

import torch
from pathlib import Path
from models.dataset import get_dataloaders
from models.cnn_lstm import CNNLSTM
from models.pose_lstm import PoseLSTM
from models.slowfast_model import get_slowfast_model
from models.videomae_model import get_videomae_model
from models.timesformer_model import get_timesformer_model
from models.resnet3d_model import get_resnet3d_model
from models.train import train_model, evaluate_model


def main():
    # Пути
    data_path = Path("./prepared_data")
    runs_path = Path("./runs")
    
    # Проверка наличия данных
    if not data_path.exists():
        print(f"❌ Ошибка: папка с данными не найдена: {data_path}")
        print("Сначала запусти prepare_data.py для подготовки данных")
        return
    
    # Загрузка данных
    print("📂 Загрузка данных...")
    train_loader, val_loader, test_loader = get_dataloaders(data_path, batch_size=16)
    print(f"  Train: {len(train_loader.dataset)} примеров")
    print(f"  Val: {len(val_loader.dataset)} примеров")
    print(f"  Test: {len(test_loader.dataset)} примеров")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Устройство: {device}")
    print("=" * 60)
    
    # ====== 1. CNN + LSTM ======
    print("\n🚀 Обучение CNN + LSTM...")
    model = CNNLSTM(num_classes=2)
    train_model(
        model, train_loader, val_loader,
        num_epochs=20, lr=5e-5, device=device,
        save_dir=runs_path / 'cnn_lstm'
    )
    evaluate_model(
        model, test_loader, device=device,
        save_dir=runs_path / 'cnn_lstm'
    )
    
    # ====== 2. 3D ResNet ======
    print("\n🚀 Обучение 3D ResNet...")
    model = get_resnet3d_model(num_classes=2)
    train_model(
        model, train_loader, val_loader,
        num_epochs=20, lr=1e-4, device=device,
        save_dir=runs_path / 'resnet_3d'
    )
    evaluate_model(
        model, test_loader, device=device,
        save_dir=runs_path / 'resnet_3d'
    )
    
    # ====== 3. SlowFast ======
    print("\n🚀 Обучение SlowFast...")
    model = get_slowfast_model(num_classes=2)
    train_model(
        model, train_loader, val_loader,
        num_epochs=20, lr=1e-4, device=device,
        save_dir=runs_path / 'slowfast'
    )
    evaluate_model(
        model, test_loader, device=device,
        save_dir=runs_path / 'slowfast'
    )
    
    # ====== 4. TimeSformer ======
    print("\n🚀 Обучение TimeSformer...")
    model = get_timesformer_model(num_classes=2)
    # TimeSformer требует специальной обработки входных данных
    # Для упрощения используем меньший batch size
    train_loader_small, val_loader_small, test_loader_small = get_dataloaders(
        data_path, batch_size=4
    )
    train_model(
        model, train_loader_small, val_loader_small,
        num_epochs=20, lr=1e-4, device=device,
        save_dir=runs_path / 'timesformer'
    )
    evaluate_model(
        model, test_loader_small, device=device,
        save_dir=runs_path / 'timesformer'
    )
    
    # ====== 5. VideoMAE ======
    print("\n🚀 Обучение VideoMAE...")
    model = get_videomae_model(num_classes=2)
    train_model(
        model, train_loader_small, val_loader_small,
        num_epochs=20, lr=1e-4, device=device,
        save_dir=runs_path / 'videomae'
    )
    evaluate_model(
        model, test_loader_small, device=device,
        save_dir=runs_path / 'videomae'
    )
    
    # ====== 6. MediaPipe + LSTM ======
    print("\n🚀 Обучение MediaPipe + LSTM...")
    # Для этой модели нужен другой датасет с извлеченными ключевыми точками
    # Это можно сделать отдельно, пока пропускаем
    
    print("\n" + "=" * 60)
    print("✅ Обучение всех моделей завершено!")
    print("📁 Результаты сохранены в папке ./runs/")


if __name__ == "__main__":
    main()