import os
from pathlib import Path
import json
from collections import defaultdict

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
    num_workers=0,
    max_batches=None,
):
    """Compute embeddings for a list of tile images."""
    dataset = TileDataset(tile_paths, preprocess=preprocess)
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
        prefetch_factor=4,
        pin_memory=True,
    )
    model.eval()
    device = next(model.parameters()).device

    embeddings = []
    wsi_ids = []

    with torch.no_grad():
        for batch_idx, (batch_images, batch_tile_ids) in enumerate(
                tqdm(dataloader, desc="Processing Tiles", unit="batch")):
            if max_batches and batch_idx >= max_batches:
                break  # Stop after processing max_batches for debugging

            batch_images = batch_images.to(device, non_blocking=True)
            embeddings.append(model(batch_images).detach().cpu().numpy())

            # Extract WSI IDs from tile IDs
            wsi_ids.extend(
                [tile_id.split("__")[0] for tile_id in batch_tile_ids])

    return np.concatenate(embeddings, axis=0), np.array(wsi_ids)


def get_hdf5_path(weights_path, output_dir):
    """Generate a single HDF5 file path based on weights filename."""
    weights_name = Path(weights_path).stem
    return Path(output_dir) / f"{weights_name}.h5"


def store_embeddings(wsi_id, embeddings, label, hdf5_path, split):
    """Store embeddings in a single HDF5 file with train/test groups."""
    with h5py.File(hdf5_path, "a") as hdf5_file:
        grp = hdf5_file.require_group(f"{split}/{wsi_id}")
        if "embeddings" in grp:
            del grp["embeddings"]  # Ensure we don't overwrite with duplicates
        grp.create_dataset("embeddings", data=embeddings, compression="gzip")
        grp.attrs["label"] = label


@click.command()
@click.option('--weights-path',
              type=click.Path(exists=True),
              help='Path to the model weights')
@click.option("--output",
              default="data/processed/embeddings/",
              help="Output directory for embeddings")
@click.option("--gpu-id", default=0, help="GPU ID to use for inference")
@click.option("--batch-size", default=256, help="")
@click.option("--num-workers", default=0, help="")
@click.option("--max-batches",
              default=None,
              type=int,
              help="Limit number of batches for debugging")
def main(
    weights_path,
    output,
    gpu_id,
    batch_size,
    num_workers,
    max_batches=None,
):
    """Precompute and store WSI embeddings in a single HDF5 file."""

    # Load metadata
    cptac_test_df = pd.read_csv(os.getenv("CPTAC_TEST_SPLIT_CSV"))
    wsi_ids_test = set(cptac_test_df["Slide_ID"].to_list())

    cptac_metadata = load_cptac_metadata(
        os.getenv("CPTAC_DATA_RAW_PATH")).set_index("Slide_ID")

    tile_paths_json = project_dir / "data/processed/metadata/cptac_tile_paths_10x.json"
    with open(tile_paths_json, "r") as f:
        tile_paths = json.load(f)

    tile_paths = [Path(p) for p in tile_paths]

    output = project_dir / output
    hdf5_path = get_hdf5_path(weights_path, output)
    hdf5_path.parent.mkdir(exist_ok=True)

    # Load model
    device = get_device(gpu_id)
    model, preprocess = load_model(weights_path, device)

    # Compute embeddings
    embeddings, wsi_ids = compute_embeddings(tile_paths,
                                             model,
                                             preprocess=preprocess,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             max_batches=max_batches)

    print("Grouping embeddings by WSI...")

    # Group embeddings by WSI ID
    wsi_embeddings = defaultdict(list)

    for embedding, wsi_id in zip(embeddings, wsi_ids):
        wsi_embeddings[wsi_id].append(embedding)

    print("Storing embeddings in HDF5...")

    # Store in HDF5
    with h5py.File(hdf5_path, "a") as hdf5_file:
        for wsi_id, emb_list in tqdm(wsi_embeddings.items(),
                                     desc="Saving WSIs",
                                     unit="WSI"):
            if wsi_id not in cptac_metadata.index:
                print(
                    f"Warning: WSI {wsi_id} not found in metadata, skipping.")
                continue

            label = cptac_metadata.loc[wsi_id, "label"]
            split = "test" if wsi_id in wsi_ids_test else "train"

            grp = hdf5_file.require_group(f"{split}/{wsi_id}")
            if "embeddings" in grp:
                del grp["embeddings"]
            grp.create_dataset("embeddings",
                               data=np.array(emb_list),
                               compression="gzip")
            grp.attrs["label"] = label

    print("All embeddings stored successfully!")


if __name__ == "__main__":
    main()
