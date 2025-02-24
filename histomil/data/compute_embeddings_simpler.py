import os
import datetime
from pathlib import Path
import json
from collections import defaultdict
from time import perf_counter

import h5py
import click
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm

from histomil.models.models import load_model, get_device
from histomil.data.torch_datasets import TileDataset

load_dotenv()

project_dir = Path(__file__).parents[2].resolve()


def load_cptac_metadata(cptac_path):
    """Load metadata for LUAD and LUSC WSIs."""
    cptac_path = Path(cptac_path)
    df_luad = pd.read_csv(cptac_path /
                          "TCIA_CPTAC_LUAD_Pathology_Data_Table.csv")
    df_lusc = pd.read_csv(cptac_path /
                          "TCIA_CPTAC_LSCC_Pathology_Data_Table.csv")

    df_luad["label"] = df_luad["Specimen_Type"].map({
        "tumor_tissue": "LUAD",
        "normal_tissue": "NORMAL"
    })
    df_lusc["label"] = df_lusc["Specimen_Type"].map({
        "tumor_tissue": "LUSC",
        "normal_tissue": "NORMAL"
    })

    return pd.concat([df_luad, df_lusc])


def compute_embeddings(
    tile_paths,
    model,
    preprocess=None,
    batch_size=128,
    num_workers=None,
    max_batches=None,
):
    """Compute embeddings dynamically based on model type and save temp checkpoints."""

    processed_tile_paths = []
    embeddings = []

    dataset = TileDataset(tile_paths, preprocess=preprocess)
    num_workers = min(4,
                      os.cpu_count() //
                      2) if num_workers is None else num_workers
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Keep workers alive
        pin_memory=True,
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"starting processing tiles with num_workers={num_workers}")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            for batch_idx, (batch_images, batch_tile_paths) in enumerate(
                    tqdm(dataloader, desc="Processing Tiles", unit="batch")):

                if max_batches and batch_idx >= max_batches:
                    break

                batch_images = batch_images.to(device, non_blocking=True)
                batch_embeddings = model(batch_images).detach().cpu().numpy()

                embeddings.extend(batch_embeddings)
                processed_tile_paths.extend(batch_tile_paths)

    embeddings = np.vstack(embeddings)
    processed_tile_paths = np.array(processed_tile_paths)
    return embeddings, processed_tile_paths


def get_npz_path(model_name, weights_path, output_dir):
    """Generate a single HDF5 file path based on weights filename."""
    if model_name == "local":
        weights_name = Path(weights_path).stem
        return Path(output_dir) / f"{weights_name}.npz"
    else:
        return Path(output_dir) / f"{model_name}.npz"


@click.command()
@click.option(
    "--model-name",
    type=str,
    default="local",
    help="Model type: 'local' for your model, 'bioptimus' for H-optimus-0")
@click.option('--weights-path',
              type=click.Path(exists=True),
              default=None,
              help='Path to the local model weights (ignored for bioptimus)')
@click.option("--output-dir",
              default="data/processed/embeddings/",
              help="Output directory for embeddings")
@click.option("--gpu-id", default=0, help="GPU ID to use for inference")
@click.option("--batch-size", default=256, help="Batch size for inference")
@click.option("--num-workers",
              default=None,
              type=click.INT,
              help="Number of workers for DataLoader")
@click.option("--max-batches",
              default=None,
              type=int,
              help="Limit number of batches for debugging")
def main(model_name,
         weights_path,
         output_dir,
         gpu_id,
         batch_size,
         num_workers,
         max_batches=None):
    """Precompute and store WSI embeddings in a single HDF5 file."""

    # Load metadata
    cptac_test_df = pd.read_csv(os.getenv("CPTAC_TEST_SPLIT_CSV"))
    wsi_ids_test = set(cptac_test_df["Slide_ID"].to_list())

    cptac_metadata = load_cptac_metadata(
        os.getenv("CPTAC_DATA_RAW_PATH")).set_index("Slide_ID")

    tile_paths_json = project_dir / "data/processed/metadata/cptac_tile_paths_10x.json"
    with open(tile_paths_json, "r") as f:
        tile_paths = json.load(f)

    total_number_tiles = len(tile_paths)

    # Load Model
    device = get_device(gpu_id)
    model, preprocess, embedding_dim = load_model(model_name, weights_path,
                                                  device)

    # Compute embeddings
    embeddings, processed_tile_paths = compute_embeddings(
        tile_paths,
        model,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        max_batches=max_batches,
    )
    output_file = get_npz_path(model_name, weights_path, output_dir)
    np.savez(output_file,
             embeddings=embeddings,
             tile_paths=processed_tile_paths)

    # Store metadata


if __name__ == "__main__":
    main()
