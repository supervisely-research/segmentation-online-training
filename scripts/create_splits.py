import os
import json
import random
import shutil
from pathlib import Path

random.seed(45)

SLY_PROJECT_PATH = "../project/"
OUTPUT_PATH = "../data"
TRAIN_RATIO = 0.8

def create_train_val_split():
    project_dir = Path(SLY_PROJECT_PATH)
    output_dir = Path(OUTPUT_PATH)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train/img").mkdir(parents=True, exist_ok=True)
    (output_dir / "train/ann").mkdir(parents=True, exist_ok=True)
    (output_dir / "val/img").mkdir(parents=True, exist_ok=True)
    (output_dir / "val/ann").mkdir(parents=True, exist_ok=True)
    
    all_items = []
    
    for dataset_dir in project_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name in ['meta.json', 'README.md']:
            continue
        
        img_dir = dataset_dir / "img"
        ann_dir = dataset_dir / "ann"
        
        if not img_dir.exists() or not ann_dir.exists():
            continue
        
        for img_file in img_dir.iterdir():
            ann_file = ann_dir / f"{img_file.name}.json"
            if ann_file.exists():
                all_items.append({
                    'img_path': img_file,
                    'ann_path': ann_file,
                    'name': f"{dataset_dir.name}_{img_file.name}"
                })
    
    random.shuffle(all_items)
    split_idx = int(len(all_items) * TRAIN_RATIO)
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]
    
    for item in train_items:
        shutil.copy2(item['img_path'], output_dir / "train/img" / item['name'])
        shutil.copy2(item['ann_path'], output_dir / "train/ann" / f"{item['name']}.json")
    
    for item in val_items:
        shutil.copy2(item['img_path'], output_dir / "val/img" / item['name'])
        shutil.copy2(item['ann_path'], output_dir / "val/ann" / f"{item['name']}.json")
    
    print(f"Train: {len(train_items)} images")
    print(f"Val: {len(val_items)} images")
    print(f"Output: {output_dir}")

if __name__ == "__main__":
    create_train_val_split()