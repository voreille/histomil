from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer

from histomil.data.torch_datasets import HDF5WSIDatasetCLAM, HDF5WSIDatasetCLAMWithTileID
from histomil.models.models import load_model, get_device
from histomil.models.clam_wrapper import PL_CLAM_SB
from histomil.visualization.heatmap import compute_attention_map

# %%
wsi_dir = Path("/mnt/nas6/data/CPTAC")

# %%
def get_wsi_path(wsi_id, wsi_dir):
    wsi_paths = [f for f in wsi_dir.rglob(wsi_id + ".svs")]
    if len(wsi_paths) > 1:
        raise ValueError(f"Multiple WSI files found for {wsi_id}: {wsi_paths}")
    return wsi_paths[0]

# %%
hdf5_path = "/home/valentin/workspaces/histomil/data/processed/embeddings/superpixels_moco_org.h5"

val_dataset = HDF5WSIDatasetCLAMWithTileID(hdf5_path, split="test")
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    num_workers=12,
    collate_fn=HDF5WSIDatasetCLAMWithTileID.get_collate_fn_ragged(),
)
test_dataset = HDF5WSIDatasetCLAM(hdf5_path, split="test")
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=12,
    collate_fn=HDF5WSIDatasetCLAM.get_collate_fn_ragged(),
)

mil_weights = "/home/valentin/workspaces/histomil/models/mil/test/clam/epoch=89-step=148860.ckpt"
device = get_device(gpu_id=1)
mil_aggregator = PL_CLAM_SB.load_from_checkpoint(mil_weights)

embeddings, labels, tile_ids = next(iter(val_loader))
mil_aggregator.to(device)
mil_aggregator.eval()   


embeddings.to(device)

pred, proba, attention_scores = mil_aggregator(embeddings)