import json
import yaml
from pathlib import Path


def create_semantic_config(
    dataset_name,
    num_classes,
    output_dir,
    base_lr=0.0001,
    max_iter=1000,
    batch_size=4,
    checkpoint_period=200
):
    """Create MaskDINO semantic segmentation config"""
    
    config = {
        '_BASE_': "/root/workspace/MaskDINO/configs/ade20k/semantic-segmentation/Base-ADE20K-SemanticSegmentation copy.yaml",
        
        'MODEL': {
            # Use only ImageNet pretrained weights, no task-specific checkpoint
            'WEIGHTS': "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
            'SEM_SEG_HEAD': {
                'NUM_CLASSES': num_classes,
            },
        },
        
        'DATASETS': {
            'TRAIN': (dataset_name,),
            'TEST': ("tomatoes_sem_seg_val",),
        },
        
        'SOLVER': {
            'IMS_PER_BATCH': batch_size,
            'BASE_LR': base_lr,
            'MAX_ITER': max_iter,
            'STEPS': (int(max_iter * 0.7), int(max_iter * 0.9)),
            'WARMUP_ITERS': min(100, max_iter // 10),
            'CHECKPOINT_PERIOD': checkpoint_period,
            'BACKBONE_MULTIPLIER': 0.1,
        },
        
        'INPUT': {
            'MIN_SIZE_TRAIN': (512,),
            'MIN_SIZE_TRAIN_SAMPLING': "choice",
            'MIN_SIZE_TEST': 512,
            'MAX_SIZE_TRAIN': 2048,
            'MAX_SIZE_TEST': 2048,
        },
        
        'TEST': {
            'EVAL_PERIOD': checkpoint_period,
        },
        
        'OUTPUT_DIR': str(output_dir),
    }
    
    return config


def generate_all_configs(data_root="/root/workspace/few_shot_data"):
    """Generate configs for all few-shot splits"""
    
    data_root = Path(data_root)
    
    # Load config - check multiple possible locations
    config_path = data_root / "config.json"
    if not config_path.exists():
        print(f"Warning: {config_path} not found")
        print("Using default configuration...")
        # Default config for tomatoes dataset
        config_data = {
            'class_to_id': {'background': 0, 'Core': 1, 'Locule': 2, 'Navel': 3, 
                           'Pericarp': 4, 'Placenta': 5, 'Septum': 6, 'Tomato': 7},
            'shots': [5, 10, 15, 30, 50],
            'num_seeds': 5
        }
    else:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    
    # Count actual classes (excluding background)
    num_classes = len([c for c in config_data['class_to_id'].values() if c > 0])
    shots = config_data['shots']
    num_seeds = config_data['num_seeds']
    
    # Create output directory for configs
    configs_dir = Path("/root/workspace/maskdino_semantic_configs")
    configs_dir.mkdir(exist_ok=True)
    
    print(f"Generating configs for {num_classes} classes")
    print(f"Shots: {shots}")
    print(f"Seeds per shot: {num_seeds}")
    print(f"Output directory: {configs_dir}\n")
    
    # Training parameters based on shot count
    training_params = {
        5: {'max_iter': 500, 'base_lr': 0.00005, 'batch_size': 2},
        10: {'max_iter': 800, 'base_lr': 0.0001, 'batch_size': 4},
        15: {'max_iter': 1000, 'base_lr': 0.0001, 'batch_size': 4},
        30: {'max_iter': 1500, 'base_lr': 0.0001, 'batch_size': 4},
        50: {'max_iter': 2000, 'base_lr': 0.0001, 'batch_size': 4},
    }
    
    generated_configs = []
    
    # Generate config for each split
    for n_shot in shots:
        for seed in range(num_seeds):
            dataset_name = f"tomatoes_sem_seg_train_{n_shot}shot_seed{seed}"
            
            # Get training parameters
            params = training_params.get(n_shot, {'max_iter': 1000, 'base_lr': 0.0001, 'batch_size': 4})
            
            # Output directory for this training run
            output_dir = Path(f"/root/workspace/outputs/maskdino_{n_shot}shot_seed{seed}")
            
            # Create config
            semantic_config = create_semantic_config(
                dataset_name=dataset_name,
                num_classes=num_classes,
                output_dir=output_dir,
                base_lr=params['base_lr'],
                max_iter=params['max_iter'],
                batch_size=params['batch_size'],
                checkpoint_period=max(100, params['max_iter'] // 5)
            )
            
            # Save config
            config_file = configs_dir / f"tomatoes_{n_shot}shot_seed{seed}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(semantic_config, f, default_flow_style=False, sort_keys=False)
            
            generated_configs.append({
                'config': str(config_file),
                'dataset': dataset_name,
                'shots': n_shot,
                'seed': seed,
                'iterations': params['max_iter']
            })
            
            print(f"✓ Generated: {config_file.name}")
    
    print(f"\n{'='*50}")
    print(f"Config generation complete!")
    print(f"Total configs: {len(generated_configs)}")
    print(f"Saved to: {configs_dir}")
    print(f"{'='*50}\n")
    
    # Create training script
    create_training_script(generated_configs, configs_dir.parent)
    
    return generated_configs


def create_training_script(configs, output_dir):
    """Create bash script for training all models"""
    
    script_content = """#!/bin/bash

# MaskDINO Semantic Segmentation Training Script
# Auto-generated script for few-shot training

set -e

cd /root/workspace/MaskDINO

echo "=========================================="
echo "MaskDINO Semantic Segmentation Training"
echo "=========================================="
echo ""

# Register datasets
echo "Registering datasets..."
python ../register_semantic_datasets.py
echo ""

"""
    
    for cfg in configs:
        config_path = cfg['config']
        dataset_name = cfg['dataset']
        n_shot = cfg['shots']
        seed = cfg['seed']
        
        script_content += f"""
echo "=========================================="
echo "Training: {n_shot}-shot seed {seed}"
echo "Dataset: {dataset_name}"
echo "Iterations: {cfg['iterations']}"
echo "=========================================="

python train_net.py \\
    --num-gpus 1 \\
    --config-file {config_path}

echo "✓ Completed: {n_shot}-shot seed {seed}"
echo ""

"""
    
    script_content += """
echo "=========================================="
echo "All training completed!"
echo "=========================================="
"""
    
    script_path = output_dir / "train_semantic_maskdino.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    import os
    os.chmod(script_path, 0o755)
    
    print(f"Training script created: {script_path}")
    print(f"Run with: bash {script_path}")


if __name__ == "__main__":
    configs = generate_all_configs()
    
    print("\nNext steps:")
    print("1. Test registration: python register_semantic_datasets.py")
    print("2. Test one config: cd MaskDINO && python train_net.py --config-file ../maskdino_semantic_configs/tomatoes_5shot_seed0.yaml --num-gpus 1")
    print("3. Run all training: bash train_semantic_maskdino.sh")