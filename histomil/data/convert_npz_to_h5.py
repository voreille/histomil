import os
import datetime
from pathlib import Path
import json
from collections import defaultdict
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor

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


def save_numpy(filename, embeddings, tile_paths):
    """Save embeddings & tile paths as NumPy arrays."""
    np.savez_compressed(filename, embeddings=embeddings, tile_paths=tile_paths)


def load_numpy(filename):
    """Load embeddings & tile paths from .npz file."""
    data = np.load(filename, allow_pickle=True)
    return list(data["embeddings"]), list(data["tile_paths"])


def save_numpy_batch(folder, batch_idx, embeddings, tile_paths):
    """Save a batch of embeddings & tile paths as separate .npz files."""
    os.makedirs(folder, exist_ok=True)  #  Ensure folder exists
    filename = os.path.join(
        folder, f"batch_{batch_idx:05d}.npz")  #  Zero-padding for sorting
    np.savez_compressed(filename, embeddings=embeddings, tile_paths=tile_paths)


def load_single_npz(filepath):
    """Load a single batch file (used for parallel loading)."""
    data = np.load(filepath, allow_pickle=True)
    return list(data["embeddings"]), list(data["tile_paths"])


def load_numpy_all(folder, parallel=True):
    """Efficiently load & merge all saved .npz batches."""
    batch_files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.endswith(".npz")
    ])

    embeddings_list = []
    tile_paths_list = []

    if parallel:
        #  Load batches in parallel (4-8 CPU cores)
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(load_single_npz, batch_files))

        # Merge results efficiently
        for embeddings, tile_paths in results:
            embeddings_list.append(embeddings)
            tile_paths_list.append(tile_paths)

    else:
        #  Sequential loading (slower but less memory overhead)
        for batch_file in batch_files:
            time_start = perf_counter()
            embeddings, tile_paths = load_single_npz(batch_file)
            embeddings_list.append(embeddings)
            tile_paths_list.append(tile_paths)
            print(f"Time taken to load one .npz: {perf_counter()-time_start}")

    return np.vstack(embeddings_list), np.concatenate(tile_paths_list)


@click.command()
@click.option("--input-dir",
              default="data/interim/embeddings/UNI2_embeddings",
              help="Directory containing the .npz to concatenate in a h5")
@click.option("--output-filepath",
              default="data/processed/embeddings/embedding.h5",
              help="Output directory for embeddings")
@click.option("--num-workers",
              default=None,
              type=click.INT,
              help="Number of workers for DataLoader")
def main(input_dir, output_filepath, num_workers):
    """Precompute and store WSI embeddings in a single HDF5 file."""
    hdf5_path = Path(output_filepath)
    hdf5_path.parent.mkdir(exist_ok=True)

    cptac_test_df = pd.read_csv(os.getenv("CPTAC_TEST_SPLIT_CSV"))
    wsi_ids_test = set(cptac_test_df["Slide_ID"].to_list())

    cptac_metadata = load_cptac_metadata(
        os.getenv("CPTAC_DATA_RAW_PATH")).set_index("Slide_ID")

    print("Storing embeddings in HDF5...")

    embeddings, processed_tile_paths = load_numpy_all(input_dir,
                                                      parallel=False)

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

    store_metadata(
        hdf5_path=hdf5_path,
        model_name="UNI",
        weights_path=None,
        embedding_dim=1536,
        total_tiles=None,
    )

    print(f"Total number of tiles (computed): {total_tiles}")
    print(f"Total number of tiles (stored as metadata): "
          f"{total_number_tiles_stored}")


if __name__ == "__main__":
    main()
