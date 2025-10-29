#!/bin/bash

# Fix nested structure in all few-shot datasets

for dataset in data/train_*shot_seed*; do
    echo "Processing: $dataset"
    
    # Fix training/images/images -> training/images
    if [ -d "$dataset/training/images/images" ]; then
        echo "  Moving images..."
        mv "$dataset/training/images/images"/* "$dataset/training/images/" 2>/dev/null
        rmdir "$dataset/training/images/images" 2>/dev/null
    fi
    
    # Fix training/labels/masks -> training/labels  
    if [ -d "$dataset/training/labels/masks" ]; then
        echo "  Moving labels..."
        mv "$dataset/training/labels/masks"/* "$dataset/training/labels/" 2>/dev/null
        rmdir "$dataset/training/labels/masks" 2>/dev/null
    fi
    
    # Fix validation/images/images -> validation/images
    if [ -d "$dataset/validation/images/images" ]; then
        echo "  Moving validation images..."
        mv "$dataset/validation/images/images"/* "$dataset/validation/images/" 2>/dev/null
        rmdir "$dataset/validation/images/images" 2>/dev/null
    fi
    
    # Fix validation/labels/masks -> validation/labels
    if [ -d "$dataset/validation/labels/masks" ]; then
        echo "  Moving validation labels..."
        mv "$dataset/validation/labels/masks"/* "$dataset/validation/labels/" 2>/dev/null
        rmdir "$dataset/validation/labels/masks" 2>/dev/null
    fi
done

echo "Done!"