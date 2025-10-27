import json
import cv2
import numpy as np
from pathlib import Path

def debug_sly_annotations():
    """Debug Supervisely annotations to see what's inside"""
    ann_dir = Path("..data/train/ann")
    
    if not ann_dir.exists():
        print("Annotation directory not found")
        return
    
    ann_files = list(ann_dir.glob("*.json"))[:5]  # Check first 5 files
    
    for ann_file in ann_files:
        print(f"\n=== {ann_file.name} ===")
        
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        
        print(f"Image size: {ann.get('size', 'not found')}")
        print(f"Objects count: {len(ann.get('objects', []))}")
        
        for i, obj in enumerate(ann.get('objects', [])):
            print(f"\nObject {i+1}:")
            print(f"  Class: {obj.get('classTitle', 'unknown')}")
            print(f"  Geometry type: {obj.get('geometryType', 'unknown')}")
            
            if 'points' in obj:
                points = obj['points']
                print(f"  Points: {points}")
                
                if obj.get('geometryType') == 'rectangle' and 'exterior' in points:
                    ext = points['exterior']
                    x1, y1 = ext[0]
                    x2, y2 = ext[1]
                    print(f"  Rectangle: ({x1},{y1}) -> ({x2},{y2})")
                    print(f"  Width: {abs(x2-x1)}, Height: {abs(y2-y1)}")
                    
                    # Check if coordinates are valid
                    img_w = ann['size']['width']
                    img_h = ann['size']['height']
                    
                    if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                        print(f"  WARNING: Negative coordinates!")
                    if x1 >= img_w or x2 >= img_w:
                        print(f"  WARNING: X coordinates exceed image width {img_w}")
                    if y1 >= img_h or y2 >= img_h:
                        print(f"  WARNING: Y coordinates exceed image height {img_h}")

def test_mask_creation():
    """Test mask creation process step by step"""
    ann_dir = Path("../data/train/ann")
    ann_files = list(ann_dir.glob("*.json"))
    
    if not ann_files:
        print("No annotation files found")
        return
    
    # Test first file with objects
    for ann_file in ann_files:
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        
        if len(ann.get('objects', [])) > 0:
            print(f"\nTesting mask creation for {ann_file.name}")
            
            img_size = (ann['size']['width'], ann['size']['height'])
            mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
            
            print(f"Created mask shape: {mask.shape}")
            print(f"Image size: {img_size}")
            
            for i, obj in enumerate(ann.get('objects', [])):
                print(f"\nProcessing object {i+1}: {obj.get('classTitle')}")
                
                if obj.get('geometryType') == 'rectangle' and 'points' in obj:
                    points = obj['points']['exterior']
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    # Ensure correct order
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    print(f"  Rectangle: ({x1},{y1}) -> ({x2},{y2})")
                    print(f"  Before: mask sum = {mask.sum()}")
                    
                    # Draw rectangle
                    if x1 >= 0 and y1 >= 0 and x2 < img_size[0] and y2 < img_size[1]:
                        mask[y1:y2, x1:x2] = 1  # Use class 1
                        print(f"  After: mask sum = {mask.sum()}")
                        print(f"  Drew rectangle successfully")
                    else:
                        print(f"  Skipped: coordinates out of bounds")
                else:
                    print(f"  Skipped: not a rectangle")
            
            # Save test mask
            test_mask_path = f"../debug_mask_{ann_file.stem}.png"
            cv2.imwrite(test_mask_path, mask * 255)  # Scale for visibility
            print(f"\nSaved test mask to {test_mask_path}")
            print(f"Final mask stats: min={mask.min()}, max={mask.max()}, sum={mask.sum()}")
            
            break  # Only test first file with objects

def check_original_project():
    """Check original Supervisely project structure"""
    project_dir = Path("../project/")
    
    if not project_dir.exists():
        print("Original project not found")
        return
    
    print("Original project structure:")
    
    # Check meta.json
    meta_path = project_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"Classes in meta: {[c['title'] for c in meta.get('classes', [])]}")
    
    # Check first dataset
    for dataset_dir in project_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name.startswith('tomato'):
            ann_dir = dataset_dir / "ann"
            if ann_dir.exists():
                ann_files = list(ann_dir.glob("*.json"))
                if ann_files:
                    ann_file = ann_files[0]
                    with open(ann_file, 'r') as f:
                        ann = json.load(f)
                    
                    print(f"\nSample from {dataset_dir.name}:")
                    print(f"  Objects: {len(ann.get('objects', []))}")
                    for obj in ann.get('objects', [])[:2]:  # First 2 objects
                        print(f"    {obj.get('classTitle')} ({obj.get('geometryType')})")
            break

if __name__ == "__main__":
    print("=== Debugging Supervisely Conversion ===")
    
    print("\n1. Checking original project...")
    check_original_project()
    
    print("\n2. Debugging split annotations...")
    debug_sly_annotations()
    
    print("\n3. Testing mask creation...")
    test_mask_creation()