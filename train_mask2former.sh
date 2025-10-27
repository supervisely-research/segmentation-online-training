#!/bin/bash

# Mask2Former Few-Shot Training Script  
# Usage: ./train_mask2former.sh

cd /workspace/Mask2Former

# Register datasets
python ../register_datasets.py


# Train 1-shot seed 0
echo "Training Mask2Former 1-shot seed 0..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_1shot_seed0.yaml


# Train 1-shot seed 1
echo "Training Mask2Former 1-shot seed 1..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_1shot_seed1.yaml


# Train 1-shot seed 2
echo "Training Mask2Former 1-shot seed 2..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_1shot_seed2.yaml


# Train 1-shot seed 3
echo "Training Mask2Former 1-shot seed 3..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_1shot_seed3.yaml


# Train 1-shot seed 4
echo "Training Mask2Former 1-shot seed 4..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_1shot_seed4.yaml


# Train 2-shot seed 0
echo "Training Mask2Former 2-shot seed 0..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_2shot_seed0.yaml


# Train 2-shot seed 1
echo "Training Mask2Former 2-shot seed 1..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_2shot_seed1.yaml


# Train 2-shot seed 2
echo "Training Mask2Former 2-shot seed 2..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_2shot_seed2.yaml


# Train 2-shot seed 3
echo "Training Mask2Former 2-shot seed 3..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_2shot_seed3.yaml


# Train 2-shot seed 4
echo "Training Mask2Former 2-shot seed 4..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_2shot_seed4.yaml


# Train 3-shot seed 0
echo "Training Mask2Former 3-shot seed 0..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_3shot_seed0.yaml


# Train 3-shot seed 1
echo "Training Mask2Former 3-shot seed 1..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_3shot_seed1.yaml


# Train 3-shot seed 2
echo "Training Mask2Former 3-shot seed 2..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_3shot_seed2.yaml


# Train 3-shot seed 3
echo "Training Mask2Former 3-shot seed 3..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_3shot_seed3.yaml


# Train 3-shot seed 4
echo "Training Mask2Former 3-shot seed 4..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_3shot_seed4.yaml


# Train 5-shot seed 0
echo "Training Mask2Former 5-shot seed 0..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_5shot_seed0.yaml


# Train 5-shot seed 1
echo "Training Mask2Former 5-shot seed 1..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_5shot_seed1.yaml


# Train 5-shot seed 2
echo "Training Mask2Former 5-shot seed 2..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_5shot_seed2.yaml


# Train 5-shot seed 3
echo "Training Mask2Former 5-shot seed 3..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_5shot_seed3.yaml


# Train 5-shot seed 4
echo "Training Mask2Former 5-shot seed 4..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_5shot_seed4.yaml


# Train 10-shot seed 0
echo "Training Mask2Former 10-shot seed 0..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_10shot_seed0.yaml


# Train 10-shot seed 1
echo "Training Mask2Former 10-shot seed 1..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_10shot_seed1.yaml


# Train 10-shot seed 2
echo "Training Mask2Former 10-shot seed 2..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_10shot_seed2.yaml


# Train 10-shot seed 3
echo "Training Mask2Former 10-shot seed 3..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_10shot_seed3.yaml


# Train 10-shot seed 4
echo "Training Mask2Former 10-shot seed 4..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_10shot_seed4.yaml


# Train 30-shot seed 0
echo "Training Mask2Former 30-shot seed 0..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_30shot_seed0.yaml


# Train 30-shot seed 1
echo "Training Mask2Former 30-shot seed 1..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_30shot_seed1.yaml


# Train 30-shot seed 2
echo "Training Mask2Former 30-shot seed 2..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_30shot_seed2.yaml


# Train 30-shot seed 3
echo "Training Mask2Former 30-shot seed 3..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_30shot_seed3.yaml


# Train 30-shot seed 4
echo "Training Mask2Former 30-shot seed 4..."
python train_net.py \
    --num-gpus 1 \
    --config-file ../mask2former_configs/tomatoes_30shot_seed4.yaml

