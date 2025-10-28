#!/bin/bash

# Script to prepare ALL few-shot splits for UNet training
# Creates data_XXX directories with proper symlinks

set -e


echo "========================================"
echo "Preparing ALL few-shot splits for UNet"
echo "========================================"

# Function to create data structure for UNet
create_unet_structure() {
    local split_name=$1
    local train_dir=$2
    local val_dir=$3
    local target_dir="data_${split_name}"
    
    echo "Creating: $target_dir"
    
    # Remove existing directory if it exists
    if [ -d "$target_dir" ]; then
        rm -rf "$target_dir"
    fi
    
    # Create directory structure
    mkdir -p "$target_dir/training" "$target_dir/validation"
    
    # Training symlinks (masks -> labels for UNet)
    ln -sf "$(realpath $train_dir/images)" "$target_dir/training/images"
    ln -sf "$(realpath $train_dir/masks)" "$target_dir/training/labels"
    
    # Validation symlinks (always use full val set)
    ln -sf "$(realpath $val_dir/images)" "$target_dir/validation/images"
    ln -sf "$(realpath $val_dir/masks)" "$target_dir/validation/labels"
    
    # Count files for verification
    local train_count=$(ls "$train_dir/images" | wc -l)
    local val_count=$(ls "$val_dir/images" | wc -l)
    
    echo "  ✓ $split_name: $train_count train + $val_count val images"
}

# Update dataset.py to use 10 classes instead of 66
echo "Updating dataset.py for 10 classes..."
sed -i 's/N_CLASSES = 66/N_CLASSES = 10/' dataset.py
echo "✓ Updated N_CLASSES = 10"

# Prepare full dataset
echo ""
echo "Preparing full dataset..."
create_unet_structure "full" "../unet_data/train" "../unet_data/val"

# Prepare all few-shot splits
echo ""
echo "Preparing few-shot splits..."

SPLITS_DIR="../few_shot_data/unet"
VAL_DIR="$SPLITS_DIR/val"

# Find all training splits
for train_split_dir in "$SPLITS_DIR"/train_*; do
    if [ -d "$train_split_dir" ]; then
        split_name=$(basename "$train_split_dir")
        create_unet_structure "$split_name" "$train_split_dir" "$VAL_DIR"
    fi
done

echo ""
echo "========================================"
echo "Summary of created datasets:"
echo "========================================"

# List all created data directories
for data_dir in data_*; do
    if [ -d "$data_dir" ]; then
        train_count=$(ls "$data_dir/training/images" | wc -l)
        val_count=$(ls "$data_dir/validation/images" | wc -l)
        echo "$data_dir: $train_count train + $val_count val"
    fi
done

echo ""
echo "All datasets prepared! You can now run:"
echo ""
echo "# Full dataset:"
echo "python train.py ./runs/full --size 512x512 --n-epochs 50"
echo ""
echo "# Few-shot examples:"
echo "python train.py ./runs/1shot_seed0 --size 512x512 --n-epochs 100"
echo "python train.py ./runs/5shot_seed0 --size 512x512 --n-epochs 100"
echo "python train.py ./runs/30shot_seed0 --size 512x512 --n-epochs 100"
echo ""

# Create convenience script for running all experiments
cat > run_all_experiments.sh << 'EOF'
#!/bin/bash

# Script to run all UNet experiments
set -e

echo "Starting UNet experiments..."

# Full dataset first
echo "Training on full dataset..."
python train.py ./runs/full --size 512x512 --n-epochs 50

# Then few-shot experiments
SHOTS=(1 2 3 5 10 30)
SEEDS=(0 1 2 3 4)

for shot in "${SHOTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        exp_name="train_${shot}shot_seed${seed}"
        data_dir="data_${exp_name}"
        
        if [ -d "$data_dir" ]; then
            echo "Training: $exp_name"
            python train.py "./runs/$exp_name" --size 512x512 --n-epochs 100
        fi
    done
done

echo "All experiments completed!"
EOF

chmod +x run_all_experiments.sh

echo "Created run_all_experiments.sh for batch training"
echo "Run: ./run_all_experiments.sh"