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
