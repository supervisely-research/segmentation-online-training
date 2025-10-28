
import cv2
import numpy as np
from pathlib import Path
import json

print('=== АНАЛИЗ КЛАССОВ В ДАННЫХ ===')

# Загружаем конфиг
with open('data_full/config.json', 'r') as f:
    config = json.load(f)

print('Классы по конфигу:')
for i, label in enumerate(config['labels']):
    print(f'  {i}: {label["name"]}')

print('\n=== СТАТИСТИКА TRAIN DATA ===')
train_masks = Path('data_full/training/labels')
class_counts_train = {}

for mask_file in train_masks.glob('*.png'):
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    unique_vals, counts = np.unique(mask, return_counts=True)
    
    for val, count in zip(unique_vals, counts):
        if val not in class_counts_train:
            class_counts_train[val] = 0
        class_counts_train[val] += count

total_pixels_train = sum(class_counts_train.values())
print('Классы в TRAIN:')
for class_id in sorted(class_counts_train.keys()):
    count = class_counts_train[class_id]
    percentage = (count / total_pixels_train) * 100
    if class_id < len(config['labels']):
        name = config['labels'][class_id]['name']
    else:
        name = 'UNKNOWN'
    print(f'  {class_id} ({name}): {count:,} пикселей ({percentage:.2f}%)')

print('\n=== ПРОБЛЕМЫ ===')
all_classes = set(range(10))
train_classes = set(class_counts_train.keys())
missing_train = all_classes - train_classes

if missing_train:
    print(f'Отсутствуют в TRAIN: {missing_train}')

# Очень редкие классы (< 0.1%)
rare_train = [cls for cls, count in class_counts_train.items() 
              if (count / total_pixels_train) < 0.001]
if rare_train:
    print(f'Очень редкие классы в TRAIN (< 0.1%): {rare_train}')

background_pct = (class_counts_train[0] / total_pixels_train) * 100
print(f'\nДИСБАЛАНС: Background составляет {background_pct:.1f}% данных')
