from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn

from histolung.evaluation.evaluators import LungHist700Evaluator
from histolung.models.models_darya import MoCoV2Encoder


def get_device(gpu_id=None):
    """Select the appropriate device for computation."""
    if torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda:0")  # Default to first GPU
    else:
        device = torch.device("cpu")
    print(f"Using {device}")
    return device


def load_model(checkpoint_path, device):
    """Load the MoCoV2 model from a given checkpoint, handling missing keys like queue_ptr."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Checkpoint {checkpoint_path} not found, skipping...")
        return None, None

    model = MoCoV2Encoder()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load the model state dict with strict=False to ignore missing keys (like queue_ptr)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    encoder = model.encoder_q
    encoder.fc = nn.Identity()

    return encoder.to(device).eval(), preprocess
