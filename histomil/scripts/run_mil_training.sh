#!/bin/bash

echo "Running MIL training..."
python histomil/training/train_mil.py \
    --hdf5-path /home/valentin/workspaces/histomil/data/interim/embeddings/UNI2_embeddings/UNI2_cptac.h5 \
    --output-name UNI2_cptac_10x_milclam \
    --gpu-id 1 \
    --batch-size 32 \
    --num-epochs 1000 \
    --num-workers 24 

