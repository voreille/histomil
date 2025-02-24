import os
import datetime
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


def store_metadata(hdf5_path, model_name, weights_path, embedding_dim,
                   total_tiles):
    """Stores metadata in the HDF5 file."""
    with h5py.File(hdf5_path, "a") as hdf5_file:
        hdf5_file.attrs["embedding_dim"] = embedding_dim
        hdf5_file.attrs["model_name"] = model_name
        hdf5_file.attrs["weights_path"] = str(
            weights_path) if weights_path else "None"
        hdf5_file.attrs["date_generated"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        hdf5_file.attrs["dataset_name"] = "CPTAC"
        hdf5_file.attrs["total_tiles"] = total_tiles
        hdf5_file.attrs["num_classes"] = 3
        hdf5_file.attrs["command"] = " ".join(os.sys.argv)


def compute_embeddings(
    tile_paths,
    model,
    preprocess=None,
    batch_size=128,
    num_workers=0,
    max_batches=None,
    save_every=100,
    temp_checkpoint=None,
):
    """Compute embeddings dynamically based on model type and save temp checkpoints."""

    #  Try to load existing checkpoint
    if temp_checkpoint and os.path.exists(temp_checkpoint):
        print(f"Resuming from {temp_checkpoint}...")
        checkpoint = np.load(temp_checkpoint, allow_pickle=True)
        processed_tile_paths = list(checkpoint["tile_paths"])
        embeddings = list(checkpoint["embeddings"])

        tile_paths = list(set(tile_paths) - set(processed_tile_paths))
    else:
        processed_tile_paths = []
        embeddings = []

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

    with torch.autocast(device_type="cuda",
                        dtype=torch.float16):  # Mixed precision for efficiency
        with torch.no_grad():
            for batch_idx, (batch_images, batch_tile_paths) in enumerate(
                    tqdm(dataloader, desc="Processing Tiles", unit="batch")):

                if max_batches and batch_idx >= max_batches:
                    break

                batch_images = batch_images.to(device, non_blocking=True)
                batch_embeddings = model(batch_images).detach().cpu().numpy()

                embeddings.extend(list(batch_embeddings))
                processed_tile_paths.extend(list(batch_tile_paths))

                #  Save checkpoint every `save_every` batches
                if (batch_idx + 1) % save_every == 0 and temp_checkpoint:
                    print(
                        f"Saving checkpoint at batch {batch_idx+1} to {temp_checkpoint}..."
                    )
                    np.savez(temp_checkpoint,
                             embeddings=np.vstack(embeddings),
                             tile_paths=np.array(processed_tile_paths))

    embeddings = np.vstack(embeddings)
    processed_tile_paths = np.array(processed_tile_paths)
    if temp_checkpoint:
        np.savez(temp_checkpoint,
                 embeddings=embeddings,
                 tile_paths=processed_tile_paths)

    return embeddings, processed_tile_paths


def get_hdf5_path(model_name, weights_path, output_dir):
    """Generate a single HDF5 file path based on weights filename."""
    if model_name == "local":
        weights_name = Path(weights_path).stem
        return Path(output_dir) / f"{weights_name}.h5"
    else:
        return Path(output_dir) / f"{model_name}.h5"


def store_embeddings(wsi_id, embeddings, label, hdf5_path, split):
    """Store embeddings in a single HDF5 file with train/test groups."""
    with h5py.File(hdf5_path, "a") as hdf5_file:
        grp = hdf5_file.require_group(f"{split}/{wsi_id}")
        if "embeddings" in grp:
            del grp["embeddings"]  # Ensure we don't overwrite with duplicates
        grp.create_dataset("embeddings", data=embeddings, compression="gzip")
        grp.attrs["label"] = label


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
              default=0,
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

    # tile_paths = [Path(p) for p in tile_paths]

    output_dir = project_dir / output_dir
    hdf5_path = get_hdf5_path(model_name, weights_path, output_dir)
    hdf5_path.parent.mkdir(exist_ok=True)

    # Define temp checkpoint path
    temp_checkpoint = project_dir / f"data/temp/{hdf5_path.stem}.npz"
    temp_checkpoint.parent.mkdir(parents=True, exist_ok=True)

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
        save_every=100,
        temp_checkpoint=temp_checkpoint,
    )

    # Store metadata
    store_metadata(hdf5_path=hdf5_path,
                   model_name=model_name,
                   weights_path=weights_path,
                   embedding_dim=embedding_dim,
                   total_tiles=len(processed_tile_paths))

    print("Storing embeddings in HDF5...")

    wsi_data = defaultdict(lambda: {"embeddings": [], "tile_ids": []})
    for tile_path, embedding in zip(processed_tile_paths, embeddings):
        tile_path_str = str(tile_path)  # Ensure it's a string
        tile_id = Path(tile_path_str).stem
        wsi_id = tile_id.split("__")[0]
        wsi_data[wsi_id]["embeddings"].append(embedding)
        wsi_data[wsi_id]["tile_ids"].append(tile_id)

    with h5py.File(hdf5_path, "a") as hdf5_file:
        for wsi_id, data in tqdm(wsi_data.items(),
                                 desc="Saving WSIs",
                                 unit="WSI"):
            label = cptac_metadata.loc[wsi_id, "label"]
            split = "test" if wsi_id in wsi_ids_test else "train"
            grp = hdf5_file.require_group(f"{split}/{wsi_id}")

            # Remove existing datasets if present
            if "embeddings" in grp:
                del grp["embeddings"]
            if "tile_ids" in grp:
                del grp["tile_ids"]

            # Store embeddings
            grp.create_dataset("embeddings",
                               data=np.array(data["embeddings"]),
                               compression="gzip")

            # Define a string dtype for storing paths
            str_dt = h5py.string_dtype(encoding='utf-8')
            grp.create_dataset("tile_ids",
                               data=np.array(data["tile_ids"], dtype=object),
                               dtype=str_dt,
                               compression="gzip")

            grp.attrs["label"] = label

    print(f"All embeddings stored successfully in {hdf5_path}!")
    print("Total number of tiles:", total_number_tiles)

    total_tiles = 0
    with h5py.File(hdf5_path, "r") as f:
        # Loop over split groups (e.g., "train" and "test")
        total_number_tiles_stored = f.attrs["total_tiles"]
        for split in f.keys():
            split_group = f[split]
            for wsi in split_group.keys():
                # Assumes that each WSI group has an "embeddings" dataset
                ds = split_group[wsi]["embeddings"]
                total_tiles += ds.shape[0]
    print("Total number of tiles (computed):", total_tiles)
    print("Total number of tiles (stored as metadata):",
          total_number_tiles_stored)

    # # Remove temp file after success
    # if temp_checkpoint.exists():
    #     temp_checkpoint.unlink()


if __name__ == "__main__":
    main()
