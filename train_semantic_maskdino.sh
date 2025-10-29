#!/bin/bash

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


echo "=========================================="
echo "Training: 5-shot seed 0"
echo "Dataset: tomatoes_sem_seg_train_5shot_seed0"
echo "Iterations: 500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_5shot_seed0.yaml

echo "✓ Completed: 5-shot seed 0"
echo ""


echo "=========================================="
echo "Training: 5-shot seed 1"
echo "Dataset: tomatoes_sem_seg_train_5shot_seed1"
echo "Iterations: 500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_5shot_seed1.yaml

echo "✓ Completed: 5-shot seed 1"
echo ""


echo "=========================================="
echo "Training: 5-shot seed 2"
echo "Dataset: tomatoes_sem_seg_train_5shot_seed2"
echo "Iterations: 500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_5shot_seed2.yaml

echo "✓ Completed: 5-shot seed 2"
echo ""


echo "=========================================="
echo "Training: 5-shot seed 3"
echo "Dataset: tomatoes_sem_seg_train_5shot_seed3"
echo "Iterations: 500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_5shot_seed3.yaml

echo "✓ Completed: 5-shot seed 3"
echo ""


echo "=========================================="
echo "Training: 5-shot seed 4"
echo "Dataset: tomatoes_sem_seg_train_5shot_seed4"
echo "Iterations: 500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_5shot_seed4.yaml

echo "✓ Completed: 5-shot seed 4"
echo ""


echo "=========================================="
echo "Training: 10-shot seed 0"
echo "Dataset: tomatoes_sem_seg_train_10shot_seed0"
echo "Iterations: 800"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_10shot_seed0.yaml

echo "✓ Completed: 10-shot seed 0"
echo ""


echo "=========================================="
echo "Training: 10-shot seed 1"
echo "Dataset: tomatoes_sem_seg_train_10shot_seed1"
echo "Iterations: 800"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_10shot_seed1.yaml

echo "✓ Completed: 10-shot seed 1"
echo ""


echo "=========================================="
echo "Training: 10-shot seed 2"
echo "Dataset: tomatoes_sem_seg_train_10shot_seed2"
echo "Iterations: 800"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_10shot_seed2.yaml

echo "✓ Completed: 10-shot seed 2"
echo ""


echo "=========================================="
echo "Training: 10-shot seed 3"
echo "Dataset: tomatoes_sem_seg_train_10shot_seed3"
echo "Iterations: 800"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_10shot_seed3.yaml

echo "✓ Completed: 10-shot seed 3"
echo ""


echo "=========================================="
echo "Training: 10-shot seed 4"
echo "Dataset: tomatoes_sem_seg_train_10shot_seed4"
echo "Iterations: 800"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_10shot_seed4.yaml

echo "✓ Completed: 10-shot seed 4"
echo ""


echo "=========================================="
echo "Training: 15-shot seed 0"
echo "Dataset: tomatoes_sem_seg_train_15shot_seed0"
echo "Iterations: 1000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_15shot_seed0.yaml

echo "✓ Completed: 15-shot seed 0"
echo ""


echo "=========================================="
echo "Training: 15-shot seed 1"
echo "Dataset: tomatoes_sem_seg_train_15shot_seed1"
echo "Iterations: 1000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_15shot_seed1.yaml

echo "✓ Completed: 15-shot seed 1"
echo ""


echo "=========================================="
echo "Training: 15-shot seed 2"
echo "Dataset: tomatoes_sem_seg_train_15shot_seed2"
echo "Iterations: 1000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_15shot_seed2.yaml

echo "✓ Completed: 15-shot seed 2"
echo ""


echo "=========================================="
echo "Training: 15-shot seed 3"
echo "Dataset: tomatoes_sem_seg_train_15shot_seed3"
echo "Iterations: 1000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_15shot_seed3.yaml

echo "✓ Completed: 15-shot seed 3"
echo ""


echo "=========================================="
echo "Training: 15-shot seed 4"
echo "Dataset: tomatoes_sem_seg_train_15shot_seed4"
echo "Iterations: 1000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_15shot_seed4.yaml

echo "✓ Completed: 15-shot seed 4"
echo ""


echo "=========================================="
echo "Training: 30-shot seed 0"
echo "Dataset: tomatoes_sem_seg_train_30shot_seed0"
echo "Iterations: 1500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_30shot_seed0.yaml

echo "✓ Completed: 30-shot seed 0"
echo ""


echo "=========================================="
echo "Training: 30-shot seed 1"
echo "Dataset: tomatoes_sem_seg_train_30shot_seed1"
echo "Iterations: 1500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_30shot_seed1.yaml

echo "✓ Completed: 30-shot seed 1"
echo ""


echo "=========================================="
echo "Training: 30-shot seed 2"
echo "Dataset: tomatoes_sem_seg_train_30shot_seed2"
echo "Iterations: 1500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_30shot_seed2.yaml

echo "✓ Completed: 30-shot seed 2"
echo ""


echo "=========================================="
echo "Training: 30-shot seed 3"
echo "Dataset: tomatoes_sem_seg_train_30shot_seed3"
echo "Iterations: 1500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_30shot_seed3.yaml

echo "✓ Completed: 30-shot seed 3"
echo ""


echo "=========================================="
echo "Training: 30-shot seed 4"
echo "Dataset: tomatoes_sem_seg_train_30shot_seed4"
echo "Iterations: 1500"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_30shot_seed4.yaml

echo "✓ Completed: 30-shot seed 4"
echo ""


echo "=========================================="
echo "Training: 50-shot seed 0"
echo "Dataset: tomatoes_sem_seg_train_50shot_seed0"
echo "Iterations: 2000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_50shot_seed0.yaml

echo "✓ Completed: 50-shot seed 0"
echo ""


echo "=========================================="
echo "Training: 50-shot seed 1"
echo "Dataset: tomatoes_sem_seg_train_50shot_seed1"
echo "Iterations: 2000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_50shot_seed1.yaml

echo "✓ Completed: 50-shot seed 1"
echo ""


echo "=========================================="
echo "Training: 50-shot seed 2"
echo "Dataset: tomatoes_sem_seg_train_50shot_seed2"
echo "Iterations: 2000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_50shot_seed2.yaml

echo "✓ Completed: 50-shot seed 2"
echo ""


echo "=========================================="
echo "Training: 50-shot seed 3"
echo "Dataset: tomatoes_sem_seg_train_50shot_seed3"
echo "Iterations: 2000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_50shot_seed3.yaml

echo "✓ Completed: 50-shot seed 3"
echo ""


echo "=========================================="
echo "Training: 50-shot seed 4"
echo "Dataset: tomatoes_sem_seg_train_50shot_seed4"
echo "Iterations: 2000"
echo "=========================================="

python train_net.py \
    --num-gpus 1 \
    --config-file /root/workspace/maskdino_semantic_configs/tomatoes_50shot_seed4.yaml

echo "✓ Completed: 50-shot seed 4"
echo ""


echo "=========================================="
echo "All training completed!"
echo "=========================================="
