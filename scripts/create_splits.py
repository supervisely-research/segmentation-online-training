import os
import json
import random
import shutil
import cv2
import numpy as np
import base64
import zlib
from pathlib import Path
from io import BytesIO
from PIL import Image
from collections import defaultdict

# Configuration
DATA_DIR = Path("../data")
PROJECT_DIR = Path("../project")  
OUTPUT_DIR = Path("../few_shot_data")
SHOTS = [1, 2, 3, 5, 10, 30]
NUM_SEEDS = 5

random.seed(42)

def load_sly_project_meta(project_path):
    """Load class information from Supervisely project meta"""
    meta_path = Path(project_path) / "meta.json"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    classes = []
    for obj_class in meta['classes']:
        classes.append(obj_class['title'])
    
    return sorted(classes)

def decode_sly_bitmap_png(bitmap_data):
    """Decode Supervisely bitmap PNG data to binary mask"""
    compressed_data = base64.b64decode(bitmap_data)
    png_bytes = zlib.decompress(compressed_data)
    img_stream = BytesIO(png_bytes)
    pil_img = Image.open(img_stream)
    bitmap_mask = np.array(pil_img)
    
    if len(bitmap_mask.shape) == 3:
        bitmap_mask = bitmap_mask[:, :, 0]
    
    return (bitmap_mask > 0).astype(np.uint8)

def create_unet_mask(ann_path, img_size, class_to_id):
    """Create UNet format mask from SLY annotation"""
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    
    mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    
    for obj in ann.get('objects', []):
        class_title = obj.get('classTitle', '')
        geom_type = obj.get('geometryType', '')
        
        if geom_type == 'bitmap' and 'bitmap' in obj:
            bitmap_info = obj['bitmap']
            if 'data' in bitmap_info:
                origin = bitmap_info.get('origin', [0, 0])
                bitmap_mask = decode_sly_bitmap_png(bitmap_info['data'])
                
                if bitmap_mask is not None:
                    class_id = class_to_id.get(class_title, 0)
                    if class_id > 0:
                        offset_x, offset_y = origin
                        h, w = bitmap_mask.shape
                        
                        y_start = max(0, offset_y)
                        x_start = max(0, offset_x)
                        y_end = min(img_size[1], offset_y + h)
                        x_end = min(img_size[0], offset_x + w)
                        
                        src_y_start = max(0, -offset_y)
                        src_x_start = max(0, -offset_x)
                        src_y_end = src_y_start + (y_end - y_start)
                        src_x_end = src_x_start + (x_end - x_start)
                        
                        if y_end > y_start and x_end > x_start:
                            final_mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
                            final_mask[y_start:y_end, x_start:x_end] = \
                                bitmap_mask[src_y_start:src_y_end, src_x_start:src_x_end]
                            
                            object_mask = (final_mask > 0).astype(np.uint8) * class_id
                            mask = np.maximum(mask, object_mask)
    
    return mask

def bitmap_to_polygon(bitmap_mask, origin):
    """Convert bitmap mask to polygon for COCO format"""
    offset_x, offset_y = origin
    
    # Find contours
    contours, _ = cv2.findContours(bitmap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0
    
    # Take largest contour
    contour = max(contours, key=cv2.contourArea)
    
    if len(contour) < 3:
        return None, 0
    
    # Convert to polygon format with offset
    polygon = []
    for point in contour:
        x, y = point[0]
        polygon.extend([float(x + offset_x), float(y + offset_y)])
    
    # Calculate area
    area = cv2.contourArea(contour)
    
    return polygon, area

def create_coco_annotations(ann_path, img_id, img_size, category_to_id):
    """Create COCO format annotations from SLY annotation"""
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    
    annotations = []
    annotation_id = 1
    
    for obj in ann.get('objects', []):
        class_title = obj.get('classTitle', '')
        geom_type = obj.get('geometryType', '')
        
        if geom_type == 'bitmap' and 'bitmap' in obj:
            bitmap_info = obj['bitmap']
            if 'data' in bitmap_info:
                origin = bitmap_info.get('origin', [0, 0])
                bitmap_mask = decode_sly_bitmap_png(bitmap_info['data'])
                
                if bitmap_mask is not None:
                    category_id = category_to_id.get(class_title)
                    if category_id is not None:
                        polygon, area = bitmap_to_polygon(bitmap_mask, origin)
                        
                        if polygon and area > 0:
                            # Calculate bounding box
                            x_coords = polygon[0::2]
                            y_coords = polygon[1::2]
                            x1, y1 = min(x_coords), min(y_coords)
                            x2, y2 = max(x_coords), max(y_coords)
                            bbox = [x1, y1, x2 - x1, y2 - y1]
                            
                            annotation = {
                                'id': annotation_id,
                                'image_id': img_id,
                                'category_id': category_id,
                                'segmentation': [polygon],
                                'area': area,
                                'bbox': bbox,
                                'iscrowd': 0
                            }
                            annotations.append(annotation)
                            annotation_id += 1
    
    return annotations

def generate_few_shot_splits():
    """Generate few-shot splits for all three model formats"""
    
    # Create output directories
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    unet_dir = output_dir / "unet"
    detectron2_dir = output_dir / "detectron2"
    images_dir = output_dir / "images"
    
    unet_dir.mkdir(exist_ok=True)
    detectron2_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # Load classes
    classes = load_sly_project_meta(PROJECT_DIR)
    class_to_id = {cls: idx + 1 for idx, cls in enumerate(classes)}
    class_to_id['background'] = 0
    
    category_to_id = {cls: idx + 1 for idx, cls in enumerate(classes)}
    coco_categories = [
        {'id': cat_id, 'name': cat_name, 'supercategory': 'none'}
        for cat_name, cat_id in category_to_id.items()
    ]
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Get all training images
    train_img_dir = DATA_DIR / "train" / "img"
    train_ann_dir = DATA_DIR / "train" / "ann"
    val_img_dir = DATA_DIR / "val" / "img"
    val_ann_dir = DATA_DIR / "val" / "ann"
    
    train_images = sorted([f.name for f in train_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    val_images = sorted([f.name for f in val_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    
    # Copy all images to shared folder
    print("Copying images to shared folder...")
    for img_name in train_images:
        shutil.copy2(train_img_dir / img_name, images_dir / f"train_{img_name}")
    
    for img_name in val_images:
        shutil.copy2(val_img_dir / img_name, images_dir / f"val_{img_name}")
    
    # Generate few-shot training splits
    for n_shot in SHOTS:
        print(f"\nGenerating {n_shot}-shot splits...")
        
        for seed in range(NUM_SEEDS):
            random.seed(42 + seed + n_shot * 100)
            
            # Sample images
            if n_shot <= len(train_images):
                sampled_images = random.sample(train_images, n_shot)
            else:
                sampled_images = train_images.copy()
            
            split_name = f"train_{n_shot}shot_seed{seed}"
            print(f"  {split_name}: {len(sampled_images)} images")
            
            # Create UNet format
            create_unet_split(sampled_images, train_img_dir, train_ann_dir, 
                            unet_dir / split_name, class_to_id)
            
            # Create COCO format for detectron2
            create_coco_split(sampled_images, train_img_dir, train_ann_dir,
                            detectron2_dir / f"{split_name}.json", 
                            coco_categories, category_to_id, "train")
    
    # Create validation splits
    print("\nGenerating validation split...")
    
    # UNet validation
    create_unet_split(val_images, val_img_dir, val_ann_dir, 
                    unet_dir / "val", class_to_id)
    
    # COCO validation
    create_coco_split(val_images, val_img_dir, val_ann_dir,
                    detectron2_dir / "val.json", 
                    coco_categories, category_to_id, "val")
    
    # Save config
    config = {
        'num_classes': len(class_to_id),
        'class_names': list(class_to_id.keys()),
        'class_to_id': class_to_id,
        'category_to_id': category_to_id,
        'classes': classes,
        'shots': SHOTS,
        'num_seeds': NUM_SEEDS
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nCompleted! Output saved to {output_dir}")
    print(f"UNet format: {unet_dir}")
    print(f"Detectron2 format: {detectron2_dir}")
    print(f"Shared images: {images_dir}")

def create_unet_split(image_list, img_dir, ann_dir, output_dir, class_to_id):
    """Create UNet format split"""
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)
    
    for img_name in image_list:
        img_path = img_dir / img_name
        ann_path = ann_dir / f"{img_name}.json"
        
        if not ann_path.exists():
            continue
        
        # Copy image
        shutil.copy2(img_path, output_dir / "images" / img_name)
        
        # Create mask
        img = cv2.imread(str(img_path))
        img_size = (img.shape[1], img.shape[0])
        
        mask = create_unet_mask(ann_path, img_size, class_to_id)
        
        mask_name = Path(img_name).stem + '.png'
        cv2.imwrite(str(output_dir / "masks" / mask_name), mask)

def create_coco_split(image_list, img_dir, ann_dir, output_file, 
                     coco_categories, category_to_id, split_prefix):
    """Create COCO format split"""
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': coco_categories
    }
    
    annotation_id = 1
    
    for img_id, img_name in enumerate(image_list, start=1):
        img_path = img_dir / img_name
        ann_path = ann_dir / f"{img_name}.json"
        
        if not ann_path.exists():
            continue
        
        # Load image info
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        
        image_info = {
            'id': img_id,
            'file_name': f"{split_prefix}_{img_name}",
            'width': width,
            'height': height
        }
        coco_data['images'].append(image_info)
        
        # Create annotations
        annotations = create_coco_annotations(ann_path, img_id, (width, height), category_to_id)
        for ann in annotations:
            ann['id'] = annotation_id
            annotation_id += 1
        
        coco_data['annotations'].extend(annotations)
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

if __name__ == "__main__":
    generate_few_shot_splits()