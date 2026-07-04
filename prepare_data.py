#!/usr/bin/env python
"""
Точка входа для запуска подготовки данных датасета GMDCSA-24

Использование:
    python prepare_data.py
"""

import sys
from pathlib import Path

# Добавляем корневую папку в PYTHONPATH
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Импортируем main из src.data.prepare
from src.data.prepare import main

if __name__ == "__main__":
    print(f"Корень проекта: {project_root}")
    print("=" * 60)
    main()