#!/bin/bash

echo "Running MIL training..."
python histomil/training/train_mil.py \
    --hdf5-path /home/valentin/workspaces/histomil/data/processed/embeddings/superpixels_moco_org.h5 \
    --output-name MOCO_ORG_mil_v1 \
    --gpu-id 1 \
    --batch-size 32 \
    --num-epochs 1000 \
    --num-workers 24 

