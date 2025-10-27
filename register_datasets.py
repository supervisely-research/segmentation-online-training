import json
from pathlib import Path
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def register_few_shot_datasets(data_dir="../few_shot_data"):
    """Register all few-shot datasets for detectron2"""
    
    data_dir = Path(data_dir)
    detectron2_dir = data_dir / "detectron2"
    images_dir = data_dir / "images"
    
    # Load config
    with open(data_dir / "config.json", 'r') as f:
        config = json.load(f)
    
    classes = config['classes']
    shots = config['shots']
    num_seeds = config['num_seeds']
    
    print(f"Registering datasets with {len(classes)} classes: {classes}")
    
    # Register training datasets
    for n_shot in shots:
        for seed in range(num_seeds):
            dataset_name = f"tomatoes_train_{n_shot}shot_seed{seed}"
            json_file = detectron2_dir / f"train_{n_shot}shot_seed{seed}.json"
            
            if json_file.exists():
                register_coco_instances(
                    dataset_name,
                    {},
                    str(json_file),
                    str(images_dir)
                )
                
                # Set metadata
                MetadataCatalog.get(dataset_name).set(
                    thing_classes=classes,
                    evaluator_type="coco"
                )
                
                print(f"Registered: {dataset_name}")
    
    # Register validation dataset
    val_json_file = detectron2_dir / "val.json"
    if val_json_file.exists():
        register_coco_instances(
            "tomatoes_val",
            {},
            str(val_json_file),
            str(images_dir)
        )
        
        MetadataCatalog.get("tomatoes_val").set(
            thing_classes=classes,
            evaluator_type="coco"
        )
        
        print("Registered: tomatoes_val")
    
    print(f"Registration complete! Available datasets:")
    print(f"  Training: tomatoes_train_{{shot}}shot_seed{{seed}} for shots={shots}, seeds=0-{num_seeds-1}")
    print(f"  Validation: tomatoes_val")

def get_dataset_info(dataset_name):
    """Get information about registered dataset"""
    from detectron2.data import DatasetCatalog
    
    try:
        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        
        print(f"\nDataset: {dataset_name}")
        print(f"  Images: {len(dataset_dicts)}")
        print(f"  Classes: {metadata.thing_classes}")
        
        # Count annotations
        total_annotations = sum(len(item.get('annotations', [])) for item in dataset_dicts)
        print(f"  Annotations: {total_annotations}")
        
        return len(dataset_dicts), total_annotations
        
    except Exception as e:
        print(f"Error accessing dataset {dataset_name}: {e}")
        return 0, 0

def verify_registration():
    """Verify that datasets are properly registered"""
    print("\nVerifying dataset registration...")
    
    # Test a few datasets
    test_datasets = [
        "tomatoes_train_1shot_seed0",
        "tomatoes_train_5shot_seed0", 
        "tomatoes_val"
    ]
    
    for dataset_name in test_datasets:
        get_dataset_info(dataset_name)

if __name__ == "__main__":
    register_few_shot_datasets()
    verify_registration()