import json
import cv2
import numpy as np
import shutil
import base64
import zlib
from pathlib import Path
from io import BytesIO
from PIL import Image

SPLITS_PATH = "../data"
OUTPUT_PATH = "../unet_data"

def load_sly_project_meta(project_path):
    meta_path = Path(project_path) / "meta.json"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    classes = []
    for obj_class in meta['classes']:
        classes.append(obj_class['title'])
    
    return sorted(classes)

def decode_sly_bitmap_png(bitmap_data, origin=None):
    """Decode Supervisely bitmap PNG data"""
    try:
        # Decode base64
        compressed_data = base64.b64decode(bitmap_data)
        
        # Decompress
        png_bytes = zlib.decompress(compressed_data)
        
        # Load as PNG image
        img_stream = BytesIO(png_bytes)
        pil_img = Image.open(img_stream)
        
        # Convert to numpy array
        bitmap_mask = np.array(pil_img)
        
        # If it's grayscale, make sure it's 2D
        if len(bitmap_mask.shape) == 3:
            bitmap_mask = bitmap_mask[:, :, 0]  # Take first channel
        
        # Convert to binary mask (non-zero = object)
        binary_mask = (bitmap_mask > 0).astype(np.uint8)
        
        print(f"    Decoded PNG: {binary_mask.shape}, {np.sum(binary_mask)} pixels")
        return binary_mask
        
    except Exception as e:
        print(f"    Error decoding PNG bitmap: {e}")
        return None

def create_mask_from_sly_bitmap(ann_path, img_size, class_to_id):
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    
    mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    objects_processed = 0
    
    print(f"Processing {ann_path.name}: {len(ann.get('objects', []))} objects")
    
    for i, obj in enumerate(ann.get('objects', [])):
        class_title = obj.get('classTitle', '')
        geom_type = obj.get('geometryType', '')
        
        print(f"  Object {i+1}: {class_title} ({geom_type})")
        
        if geom_type == 'bitmap' and 'bitmap' in obj:
            bitmap_info = obj['bitmap']
            if 'data' in bitmap_info:
                origin = bitmap_info.get('origin', [0, 0])
                
                # Decode PNG bitmap
                bitmap_mask = decode_sly_bitmap_png(bitmap_info['data'], origin)
                
                if bitmap_mask is not None:
                    class_id = class_to_id.get(class_title, 0)
                    if class_id > 0:
                        # Apply origin offset
                        offset_x, offset_y = origin
                        
                        # Create final mask with proper positioning
                        final_mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
                        
                        h, w = bitmap_mask.shape
                        
                        # Calculate bounds
                        y_start = max(0, offset_y)
                        x_start = max(0, offset_x)
                        y_end = min(img_size[1], offset_y + h)
                        x_end = min(img_size[0], offset_x + w)
                        
                        # Calculate source bounds
                        src_y_start = max(0, -offset_y)
                        src_x_start = max(0, -offset_x)
                        src_y_end = src_y_start + (y_end - y_start)
                        src_x_end = src_x_start + (x_end - x_start)
                        
                        if y_end > y_start and x_end > x_start:
                            # Place bitmap at correct position
                            final_mask[y_start:y_end, x_start:x_end] = \
                                bitmap_mask[src_y_start:src_y_end, src_x_start:src_x_end]
                            
                            # Apply class ID
                            object_mask = (final_mask > 0).astype(np.uint8) * class_id
                            
                            # Add to main mask (handle overlaps by taking max)
                            mask = np.maximum(mask, object_mask)
                            
                            objects_processed += 1
                            pixels_count = np.sum(final_mask > 0)
                            print(f"    ✓ Drew class {class_id} ({class_title}): {pixels_count} pixels at ({offset_x},{offset_y})")
                        else:
                            print(f"    Object outside image bounds")
                    else:
                        print(f"    Unknown class: {class_title}")
                else:
                    print(f"    Failed to decode PNG bitmap")
            else:
                print(f"    No bitmap data")
        else:
            print(f"    Unsupported geometry: {geom_type}")
    
    print(f"  Total objects processed: {objects_processed}")
    return mask

def sly_to_unet_png():
    splits_dir = Path(SPLITS_PATH)
    output_dir = Path(OUTPUT_PATH)
    
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Load classes from original project
    try:
        classes = load_sly_project_meta("../project")
        print(f"Found classes: {classes}")
    except Exception as e:
        print(f"Could not load meta: {e}")
        return
    
    class_to_id = {cls: idx + 1 for idx, cls in enumerate(classes)}
    class_to_id['background'] = 0
    
    print(f"Class mapping: {class_to_id}")
    
    total_processed = 0
    total_with_objects = 0
    
    for split in ['train', 'val']:
        print(f"\nProcessing {split} split...")
        
        img_dir = splits_dir / split / 'img'
        ann_dir = splits_dir / split / 'ann'
        
        if not img_dir.exists() or not ann_dir.exists():
            print(f"  Split directories not found: {img_dir}, {ann_dir}")
            continue
        
        out_img_dir = output_dir / split / 'images'
        out_mask_dir = output_dir / split / 'masks'
        
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            ann_file = ann_dir / f"{img_file.name}.json"
            if not ann_file.exists():
                print(f"No annotation for {img_file.name}")
                continue
            
            # Copy image
            shutil.copy2(img_file, out_img_dir / img_file.name)
            
            # Create mask
            img = cv2.imread(str(img_file))
            img_size = (img.shape[1], img.shape[0])  # (width, height)
            
            mask = create_mask_from_sly_bitmap(ann_file, img_size, class_to_id)
            
            mask_name = img_file.stem + '.png'
            cv2.imwrite(str(out_mask_dir / mask_name), mask)
            
            total_processed += 1
            unique_values = np.unique(mask)
            if len(unique_values) > 1 or mask.max() > 0:
                total_with_objects += 1
                print(f"  ✓ Saved mask with objects: {mask_name} (classes: {unique_values})")
            else:
                print(f"  ✗ Saved empty mask: {mask_name}")
            
    
    # Save config
    config = {
        'num_classes': len(class_to_id),
        'class_names': list(class_to_id.keys()),
        'class_to_id': class_to_id
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"PNG UNET data saved to {output_dir}")
    print(f"Classes: {len(class_to_id)} - {list(class_to_id.keys())}")
    print(f"Total images processed: {total_processed}")
    print(f"Images with objects: {total_with_objects}")
    print(f"Images with only background: {total_processed - total_with_objects}")

if __name__ == "__main__":
    sly_to_unet_png()