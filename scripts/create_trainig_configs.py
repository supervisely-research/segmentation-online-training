import json
import yaml
from pathlib import Path

def create_maskdino_config(dataset_name, num_classes, output_dir, base_lr=0.0001, max_iter=1000):
    """Create MaskDINO config for few-shot training"""
    
    config = {
        '_BASE_': "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
        
        # Dataset settings
        'DATASETS': {
            'TRAIN': (dataset_name,),
            'TEST': ("tomatoes_val",)
        },
        
        # Model settings
        'MODEL': {
            'WEIGHTS': "model_zoo://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
            'SEM_SEG_HEAD': {
                'NUM_CLASSES': num_classes
            },
            'ROI_HEADS': {
                'NUM_CLASSES': num_classes
            }
        },
        
        # Solver settings
        'SOLVER': {
            'IMS_PER_BATCH': 2,
            'BASE_LR': base_lr,
            'MAX_ITER': max_iter,
            'STEPS': [int(max_iter * 0.7), int(max_iter * 0.9)],
            'WARMUP_ITERS': min(100, max_iter // 10),
            'CHECKPOINT_PERIOD': max_iter // 5
        },
        
        # Input settings
        'INPUT': {
            'MIN_SIZE_TRAIN': (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
            'MIN_SIZE_TEST': 800,
            'MAX_SIZE_TRAIN': 1333,
            'MAX_SIZE_TEST': 1333,
        },
        
        # Output
        'OUTPUT_DIR': str(output_dir)
    }
    
    return config

def create_mask2former_config(dataset_name, num_classes, output_dir, base_lr=0.0001, max_iter=1000):
    """Create Mask2Former config for few-shot training"""
    
    config = {
        '_BASE_': "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        
        # Dataset settings
        'DATASETS': {
            'TRAIN': (dataset_name,),
            'TEST': ("tomatoes_val",)
        },
        
        # Model settings
        'MODEL': {
            'WEIGHTS': "model_zoo://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
            'SEM_SEG_HEAD': {
                'NUM_CLASSES': num_classes
            },
            'MASK_FORMER': {
                'NUM_OBJECT_QUERIES': 100,
                'TRANSFORMER_DECODER_NAME': "MultiScaleMaskedTransformerDecoder",
                'TRANSFORMER_IN_FEATURE': "multi_scale_pixel_decoder",
                'DEEP_SUPERVISION': True,
                'NO_OBJECT_WEIGHT': 0.1,
                'CLASS_WEIGHT': 2.0,
                'MASK_WEIGHT': 5.0,
                'DICE_WEIGHT': 5.0,
                'HIDDEN_DIM': 256,
                'NUM_OBJECT_QUERIES': 100,
                'NHEADS': 8,
                'DROPOUT': 0.0,
                'DIM_FEEDFORWARD': 2048,
                'ENC_LAYERS': 0,
                'DEC_LAYERS': 10,
                'PRE_NORM': False,
                'ENFORCE_INPUT_PROJ': False,
                'SIZE_DIVISIBILITY': 32,
                'DEC_N_POINTS': 4,
                'ENC_N_POINTS': 4,
                'TRAIN_NUM_POINTS': 12544,
                'OVERSAMPLE_RATIO': 3.0,
                'IMPORTANCE_SAMPLE_RATIO': 0.75,
            }
        },
        
        # Solver settings
        'SOLVER': {
            'IMS_PER_BATCH': 2,
            'BASE_LR': base_lr,
            'MAX_ITER': max_iter,
            'STEPS': [int(max_iter * 0.7), int(max_iter * 0.9)],
            'WARMUP_ITERS': min(100, max_iter // 10),
            'CHECKPOINT_PERIOD': max_iter // 5,
            'OPTIMIZER': "ADAMW",
            'WEIGHT_DECAY': 0.05
        },
        
        # Input settings
        'INPUT': {
            'MIN_SIZE_TRAIN': (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
            'MIN_SIZE_TEST': 800,
            'MAX_SIZE_TRAIN': 1333,
            'MAX_SIZE_TEST': 1333,
        },
        
        # Output
        'OUTPUT_DIR': str(output_dir)
    }
    
    return config

def generate_training_configs(data_dir="../few_shot_data"):
    """Generate training configs for all few-shot datasets"""
    
    data_dir = Path(data_dir)
    
    # Load config
    with open(data_dir / "config.json", 'r') as f:
        config = json.load(f)
    
    classes = config['classes']
    shots = config['shots']
    num_seeds = config['num_seeds']
    num_classes = len(classes)
    
    # Create config directories
    maskdino_configs_dir = Path("../maskdino_configs")
    mask2former_configs_dir = Path("../mask2former_configs")
    
    maskdino_configs_dir.mkdir(exist_ok=True)
    mask2former_configs_dir.mkdir(exist_ok=True)
    
    print(f"Generating configs for {num_classes} classes: {classes}")
    
    # Generate configs for each few-shot setting
    for n_shot in shots:
        for seed in range(num_seeds):
            dataset_name = f"tomatoes_train_{n_shot}shot_seed{seed}"
            
            # Adjust training parameters based on shot count
            if n_shot <= 5:
                max_iter = 500
                base_lr = 0.00005
            elif n_shot <= 10:
                max_iter = 800
                base_lr = 0.0001
            else:
                max_iter = 1200
                base_lr = 0.0001
            
            # MaskDINO config
            maskdino_output = Path(f"../outputs/maskdino_{n_shot}shot_seed{seed}")
            maskdino_config = create_maskdino_config(
                dataset_name, num_classes, maskdino_output, base_lr, max_iter
            )
            
            config_file = maskdino_configs_dir / f"tomatoes_{n_shot}shot_seed{seed}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(maskdino_config, f, default_flow_style=False)
            
            # Mask2Former config
            mask2former_output = Path(f"../outputs/mask2former_{n_shot}shot_seed{seed}")
            mask2former_config = create_mask2former_config(
                dataset_name, num_classes, mask2former_output, base_lr, max_iter
            )
            
            config_file = mask2former_configs_dir / f"tomatoes_{n_shot}shot_seed{seed}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(mask2former_config, f, default_flow_style=False)
            
            print(f"Generated configs for {dataset_name}")
    
    # Create training scripts
    create_training_scripts(shots, num_seeds)
    
    print(f"\nConfigs saved to:")
    print(f"  MaskDINO: {maskdino_configs_dir}")
    print(f"  Mask2Former: {mask2former_configs_dir}")

def create_training_scripts(shots, num_seeds):
    """Create training scripts for batch execution"""
    
    # MaskDINO training script
    maskdino_script = """#!/bin/bash

# MaskDINO Few-Shot Training Script
# Usage: ./train_maskdino.sh

cd /workspace/MaskDINO

# Register datasets
python ../register_datasets.py

"""
    
    for n_shot in shots:
        for seed in range(num_seeds):
            maskdino_script += f"""
# Train {n_shot}-shot seed {seed}
echo "Training MaskDINO {n_shot}-shot seed {seed}..."
python train_net.py \\
    --num-gpus 1 \\
    --config-file ../maskdino_configs/tomatoes_{n_shot}shot_seed{seed}.yaml

"""
    
    with open("../train_maskdino.sh", 'w') as f:
        f.write(maskdino_script)
    
    # Mask2Former training script
    mask2former_script = """#!/bin/bash

# Mask2Former Few-Shot Training Script  
# Usage: ./train_mask2former.sh

cd /workspace/Mask2Former

# Register datasets
python ../register_datasets.py

"""
    
    for n_shot in shots:
        for seed in range(num_seeds):
            mask2former_script += f"""
# Train {n_shot}-shot seed {seed}
echo "Training Mask2Former {n_shot}-shot seed {seed}..."
python train_net.py \\
    --num-gpus 1 \\
    --config-file ../mask2former_configs/tomatoes_{n_shot}shot_seed{seed}.yaml

"""
    
    with open("../train_mask2former.sh", 'w') as f:
        f.write(mask2former_script)
    
    # Make scripts executable
    import os
    os.chmod("../train_maskdino.sh", 0o755)
    os.chmod("../train_mask2former.sh", 0o755)
    
    print("\nTraining scripts created:")
    print("  ../train_maskdino.sh")
    print("  ../train_mask2former.sh")

def create_unet_training_configs(data_dir="../few_shot_data"):
    """Create UNet training configuration"""
    
    data_dir = Path(data_dir)
    
    # Load config
    with open(data_dir / "config.json", 'r') as f:
        config = json.load(f)
    
    classes = config['classes']
    shots = config['shots']
    num_seeds = config['num_seeds']
    num_classes = len(classes) + 1  # +1 for background
    
    unet_config = {
        'model': {
            'num_classes': num_classes,
            'class_names': ['background'] + classes
        },
        'data': {
            'train_splits': [],
            'val_split': str(data_dir / "unet" / "val"),
            'image_size': [512, 512],
            'augmentation': True
        },
        'training': {
            'batch_size': 4,
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'loss': 'crossentropy'
        }
    }
    
    # Add all training splits
    for n_shot in shots:
        for seed in range(num_seeds):
            split_name = f"train_{n_shot}shot_seed{seed}"
            unet_config['data']['train_splits'].append({
                'name': split_name,
                'path': str(data_dir / "unet" / split_name),
                'shots': n_shot,
                'seed': seed
            })
    
    # Save UNet config
    with open("../unet_config.json", 'w') as f:
        json.dump(unet_config, f, indent=2)
    
    print(f"UNet config saved to: ../unet_config.json")

if __name__ == "__main__":
    generate_training_configs()
    create_unet_training_configs()