import json
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


def load_tomato_sem_seg(image_dir, gt_dir):
    """Load semantic segmentation dataset with images and PNG masks"""
    image_dir = Path(image_dir)
    gt_dir = Path(gt_dir)
    
    dataset_dicts = []
    
    # Get all image files
    image_files = sorted(list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")))
    
    for img_file in image_files:
        # Find corresponding mask
        mask_file = gt_dir / f"{img_file.stem}.png"
        
        if not mask_file.exists():
            print(f"Warning: mask not found for {img_file.name}")
            continue
        
        record = {
            "file_name": str(img_file),
            "sem_seg_file_name": str(mask_file),
        }
        dataset_dicts.append(record)
    
    return dataset_dicts


def register_tomato_semantic_datasets(data_root="/root/workspace/few_shot_data"):
    """Register all few-shot semantic segmentation datasets"""
    
    data_root = Path(data_root)
    unet_dir = data_root / "unet"
    
    # Load config
    with open(data_root / "config.json", 'r') as f:
        config = json.load(f)
    
    # Get class names (excluding background as it's index 0)
    class_names = config['class_names']
    num_classes = len([c for c in config['class_to_id'].values() if c > 0])  # Count non-background classes
    
    # Semantic segmentation uses stuff_classes (all classes including background)
    stuff_classes = class_names
    
    print(f"Registering datasets with {num_classes} classes (+ background)")
    print(f"Classes: {stuff_classes}")
    
    # Register validation dataset
    val_images = unet_dir / "val" / "images"
    val_masks = unet_dir / "val" / "masks"
    
    if val_images.exists() and val_masks.exists():
        from functools import partial
        DatasetCatalog.register(
            "tomatoes_sem_seg_val",
            partial(load_tomato_sem_seg, str(val_images), str(val_masks))
        )
        
        MetadataCatalog.get("tomatoes_sem_seg_val").set(
            stuff_classes=stuff_classes,
            stuff_colors=None,  # Will use default colors
            ignore_label=255,
            evaluator_type="sem_seg",
        )
        
        val_count = len(list(val_images.glob("*.jpeg")) + list(val_images.glob("*.jpg")) + list(val_images.glob("*.JPG")))
        print(f"✓ Registered: tomatoes_sem_seg_val ({val_count} images)")
    
    # Register training datasets for all splits
    registered_count = 0
    
    for split_dir in sorted(unet_dir.glob("train_*")):
        if not split_dir.is_dir():
            continue
        
        split_name = split_dir.name
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            print(f"Warning: skipping {split_name} - missing images or masks")
            continue
        
        dataset_name = f"tomatoes_sem_seg_{split_name}"
        
        # Use functools.partial to avoid lambda scope issues
        from functools import partial
        DatasetCatalog.register(
            dataset_name,
            partial(load_tomato_sem_seg, str(images_dir), str(masks_dir))
        )
        
        MetadataCatalog.get(dataset_name).set(
            stuff_classes=stuff_classes,
            stuff_colors=None,
            ignore_label=255,
            evaluator_type="sem_seg",
        )
        
        img_count = len(list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")))
        print(f"✓ Registered: {dataset_name} ({img_count} images)")
        registered_count += 1
    
    print(f"\n{'='*50}")
    print(f"Registration complete!")
    print(f"Total training splits: {registered_count}")
    print(f"Validation split: 1")
    print(f"{'='*50}\n")
    
    # Print example usage
    print("Available datasets:")
    print("  Training: tomatoes_sem_seg_train_{shot}shot_seed{seed}")
    print("    Example: tomatoes_sem_seg_train_5shot_seed0")
    print("  Validation: tomatoes_sem_seg_val")


if __name__ == "__main__":
    register_tomato_semantic_datasets()
    
    # Verify one dataset
    print("\nVerifying registration...")
    try:
        from detectron2.data import DatasetCatalog
        dataset = DatasetCatalog.get("tomatoes_sem_seg_train_5shot_seed0")
        print(f"✓ Successfully loaded tomatoes_sem_seg_train_5shot_seed0: {len(dataset)} samples")
        print(f"  First sample: {dataset[0]['file_name']}")
    except Exception as e:
        print(f"✗ Error: {e}")