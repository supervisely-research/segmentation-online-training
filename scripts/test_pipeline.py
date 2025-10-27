import json
import cv2
import numpy as np
from pathlib import Path

def test_few_shot_pipeline():
    """Test the complete few-shot data preparation pipeline"""
    
    print("="*60)
    print("TESTING FEW-SHOT DATA PREPARATION PIPELINE")
    print("="*60)
    
    # Check if output directory exists
    output_dir = Path("../few_shot_data")
    if not output_dir.exists():
        print("❌ Output directory not found. Run create_few_shot_splits.py first.")
        return False
    
    # Load config
    config_file = output_dir / "config.json"
    if not config_file.exists():
        print("❌ Config file not found.")
        return False
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    classes = config['classes']
    shots = config['shots']
    num_seeds = config['num_seeds']
    
    print(f"✓ Config loaded: {len(classes)} classes, shots={shots}, seeds={num_seeds}")
    print(f"  Classes: {classes}")
    
    # Test directory structure
    unet_dir = output_dir / "unet"
    detectron2_dir = output_dir / "detectron2"
    images_dir = output_dir / "images"
    
    if not all([unet_dir.exists(), detectron2_dir.exists(), images_dir.exists()]):
        print("❌ Missing output directories")
        return False
    
    print("✓ Output directories exist")
    
    # Test shared images
    train_images = list(images_dir.glob("train_*"))
    val_images = list(images_dir.glob("val_*"))
    
    print(f"✓ Shared images: {len(train_images)} train, {len(val_images)} val")
    
    # Test UNet format
    print("\n📁 Testing UNet format...")
    
    # Check validation split
    val_dir = unet_dir / "val"
    if val_dir.exists():
        val_images_unet = list((val_dir / "images").glob("*"))
        val_masks_unet = list((val_dir / "masks").glob("*"))
        print(f"  ✓ Val split: {len(val_images_unet)} images, {len(val_masks_unet)} masks")
    else:
        print("  ❌ Val split missing")
        return False
    
    # Check a few training splits
    test_splits = ["train_1shot_seed0", "train_5shot_seed0", "train_30shot_seed0"]
    
    for split_name in test_splits:
        split_dir = unet_dir / split_name
        if split_dir.exists():
            images = list((split_dir / "images").glob("*"))
            masks = list((split_dir / "masks").glob("*"))
            print(f"  ✓ {split_name}: {len(images)} images, {len(masks)} masks")
            
            # Test mask values
            if masks:
                mask_path = masks[0]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                unique_values = np.unique(mask)
                print(f"    Mask values: {unique_values}")
        else:
            print(f"  ❌ {split_name} missing")
    
    # Test COCO format
    print("\n📁 Testing COCO format...")
    
    # Check validation
    val_json = detectron2_dir / "val.json"
    if val_json.exists():
        with open(val_json, 'r') as f:
            val_data = json.load(f)
        
        print(f"  ✓ Val JSON: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
        print(f"    Categories: {len(val_data['categories'])}")
    else:
        print("  ❌ Val JSON missing")
        return False
    
    # Check training splits
    for split_name in test_splits:
        json_file = detectron2_dir / f"{split_name}.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"  ✓ {split_name}.json: {len(data['images'])} images, {len(data['annotations'])} annotations")
        else:
            print(f"  ❌ {split_name}.json missing")
    
    # Test annotation consistency
    print("\n🔍 Testing annotation consistency...")
    
    test_split = "train_5shot_seed0"
    
    # UNet split
    unet_split_dir = unet_dir / test_split
    unet_images = sorted((unet_split_dir / "images").glob("*"))
    
    # COCO split
    coco_json = detectron2_dir / f"{test_split}.json"
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    coco_images = sorted([img['file_name'] for img in coco_data['images']])
    
    # Remove prefixes for comparison
    unet_names = [img.name for img in unet_images]
    coco_names = [name.replace('train_', '') for name in coco_images]
    
    if set(unet_names) == set(coco_names):
        print(f"  ✓ Image consistency check passed for {test_split}")
        print(f"    UNet: {len(unet_names)} images")
        print(f"    COCO: {len(coco_names)} images")
    else:
        print(f"  ❌ Image consistency check failed for {test_split}")
        print(f"    UNet only: {set(unet_names) - set(coco_names)}")
        print(f"    COCO only: {set(coco_names) - set(unet_names)}")
        return False
    
    # Test sample annotation quality
    print("\n🔍 Testing annotation quality...")
    
    # Load first image from val split and check mask
    if val_images_unet:
        sample_img_path = val_images_unet[0]
        sample_mask_path = val_dir / "masks" / f"{sample_img_path.stem}.png"
        
        if sample_mask_path.exists():
            img = cv2.imread(str(sample_img_path))
            mask = cv2.imread(str(sample_mask_path), cv2.IMREAD_GRAYSCALE)
            
            print(f"  ✓ Sample image: {img.shape} ({sample_img_path.name})")
            print(f"  ✓ Sample mask: {mask.shape} (values: {np.unique(mask)})")
            
            # Check if mask has objects
            if mask.max() > 0:
                print(f"    Objects found: {np.sum(mask > 0)} pixels")
            else:
                print(f"    ⚠️  Empty mask")
        else:
            print(f"  ❌ Sample mask not found")
    
    # Check COCO annotation sample
    if val_data['annotations']:
        sample_ann = val_data['annotations'][0]
        print(f"  ✓ Sample COCO annotation:")
        print(f"    Category: {sample_ann['category_id']}")
        print(f"    BBox: {sample_ann['bbox']}")
        print(f"    Area: {sample_ann['area']}")
        print(f"    Segmentation: {len(sample_ann['segmentation'][0])} points")
    
    print("\n" + "="*60)
    print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"✓ UNet data ready in: {unet_dir}")
    print(f"✓ COCO data ready in: {detectron2_dir}")
    print(f"✓ Shared images in: {images_dir}")
    print("\nNext steps:")
    print("1. Run register_datasets.py to register COCO datasets")
    print("2. Run create_training_configs.py to generate training configs")
    print("3. Start training with generated scripts")
    
    return True

def quick_stats():
    """Print quick statistics about the generated data"""
    
    output_dir = Path("../few_shot_data")
    if not output_dir.exists():
        print("No data found. Run create_few_shot_splits.py first.")
        return
    
    config_file = output_dir / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    shots = config['shots']
    num_seeds = config['num_seeds']
    
    print("\n📊 QUICK STATISTICS")
    print("-" * 40)
    
    # Count UNet splits
    unet_dir = output_dir / "unet"
    unet_splits = len([d for d in unet_dir.iterdir() if d.is_dir()])
    print(f"UNet splits: {unet_splits}")
    
    # Count COCO files
    detectron2_dir = output_dir / "detectron2"
    coco_files = len(list(detectron2_dir.glob("*.json")))
    print(f"COCO files: {coco_files}")
    
    # Count images
    images_dir = output_dir / "images"
    total_images = len(list(images_dir.glob("*")))
    print(f"Total images: {total_images}")
    
    # Expected counts
    expected_unet = len(shots) * num_seeds + 1  # +1 for val
    expected_coco = len(shots) * num_seeds + 1  # +1 for val
    
    print(f"Expected UNet splits: {expected_unet}")
    print(f"Expected COCO files: {expected_coco}")
    
    if unet_splits == expected_unet and coco_files == expected_coco:
        print("✅ All files present!")
    else:
        print("⚠️  Some files missing")

if __name__ == "__main__":
    success = test_few_shot_pipeline()
    if success:
        quick_stats()