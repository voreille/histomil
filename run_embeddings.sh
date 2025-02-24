#!/bin/bash

echo "Running first model..."
python histomil/data/compute_embeddings.py --weights-path /mnt/nas7/data/Personal/Darya/saved_models/superpixel_org/superpixel_org_99.pth --num-workers 32 --gpu-id 1 --batch-size 512

echo "First model done. Running Bioptimus..."
python histomil/data/compute_embeddings.py --weights-path models/test/superpixel_moco_org_99.pth --num-workers 32 --gpu-id 1 --batch-size 512

echo "All embeddings computed successfully!"
