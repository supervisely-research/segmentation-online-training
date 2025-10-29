#!/bin/bash

set -e

echo "========================================"
echo "Migrating new few-shot data to unet/data"
echo "========================================"

SOURCE_DIR="../few_shot_data/unet"
TARGET_DIR="./data"
VAL_SOURCE="${SOURCE_DIR}/val"

# Backup old config if exists
if [ -f "${TARGET_DIR}/config.json" ]; then
    echo "Backing up old config..."
    cp "${TARGET_DIR}/config.json" "${TARGET_DIR}/config.json.old"
fi

# Clean old data
echo "Removing old data..."
rm -rf ${TARGET_DIR}/*

# Copy new config
echo "Copying new config..."
cp ../few_shot_data/config.json ${TARGET_DIR}/

# Function to adapt structure for UNet format
adapt_split() {
    local split_name=$1
    local source_path="${SOURCE_DIR}/${split_name}"
    local target_path="${TARGET_DIR}/${split_name}"
    
    echo "Processing: ${split_name}"
    
    # Create structure
    mkdir -p "${target_path}/training"
    mkdir -p "${target_path}/validation"
    
    # Training: symlink images and masks->labels
    ln -sf "$(realpath ${source_path}/images)" "${target_path}/training/images"
    ln -sf "$(realpath ${source_path}/masks)" "${target_path}/training/labels"
    
    # Validation: always use shared val split
    ln -sf "$(realpath ${VAL_SOURCE}/images)" "${target_path}/validation/images"
    ln -sf "$(realpath ${VAL_SOURCE}/masks)" "${target_path}/validation/labels"
    
    # Count files
    local train_count=$(ls ${source_path}/images | wc -l)
    local val_count=$(ls ${VAL_SOURCE}/images | wc -l)
    
    echo "  ✓ ${split_name}: ${train_count} train + ${val_count} val"
}

# Process val split
echo ""
echo "Creating validation split..."
adapt_split "val"

# Process all training splits
echo ""
echo "Creating training splits..."
for split_dir in ${SOURCE_DIR}/train_*; do
    if [ -d "$split_dir" ]; then
        split_name=$(basename "$split_dir")
        adapt_split "$split_name"
    fi
done

echo ""
echo "========================================"
echo "Migration completed!"
echo "========================================"
echo ""
echo "Summary:"
ls -1 ${TARGET_DIR} | grep -v ".json" | wc -l | xargs echo "Total splits:"

echo ""
echo "Next steps:"
echo "1. Update dataset.py: N_CLASSES = 8"
echo "2. Run training: python train.py ./runs/test --size 512x512 --n-epochs 10"