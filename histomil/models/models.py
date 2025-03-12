from pathlib import Path
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import timm
from huggingface_hub import login
from dotenv import load_dotenv
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from histolung.models.models_darya import MoCoV2Encoder

load_dotenv()


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


def load_local_model(checkpoint_path, device):
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

    encoder = model.encoder_q
    encoder.fc = nn.Identity()

    return encoder.to(device).eval()


def load_model(model_name, weights_path, device):
    """Load the model dynamically based on the model name."""

    if model_name == "bioptimus":
        login(token=os.getenv("HUGGING_FACE_TOKEN"))
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0",
                                  pretrained=True,
                                  init_values=1e-5,
                                  dynamic_img_size=False)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.707223, 0.578729, 0.703617),
                                 std=(0.211883, 0.230117, 0.177517)),
        ])
        embedding_dim = 1536

    elif model_name == "UNI2":
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h",
                                  pretrained=True,
                                  **timm_kwargs)
        preprocess = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model))
        embedding_dim = 1536
    else:
        # Load your custom model from local weights
        model = load_local_model(weights_path,
                                 device)  # Your existing function
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        embedding_dim = 2048

    model.to(device)
    model.eval()
    return model, preprocess, embedding_dim
