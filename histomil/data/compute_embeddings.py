# Define label mapping (ensure consistent order)
import os
from pathlib import Path

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
):
    """Compute embeddings for a list of tile images."""
    dataset = TileDataset(tile_paths, preprocess=preprocess)
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
    )
    model.eval()
    device = next(model.parameters()).device  # Get the device of the model's parameters

    embeddings = []
    with torch.no_grad():
        for image in dataloader:
            image = image.to(device, non_blocking=True)
            embeddings.append(model(image).detach().to("cpu").numpy())

    return np.concatenate(embeddings, axis=0)


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
@click.option("--batch-size", default=128, help="")
@click.option("--num-workers", default=0, help="")
def main(weights_path, output, gpu_id, batch_size, num_workers):
    """Precompute and store WSI embeddings in a single HDF5 file."""

    # Load metadata and test/train split
    cptac_test_df = pd.read_csv(os.getenv("CPTAC_TEST_SPLIT_CSV"))
    wsi_ids_test = set(cptac_test_df["Slide_ID"].to_list())
    tiles_basedir = Path(os.getenv("TILES_10X_BASEDIR"))
    cptac_metadata = load_cptac_metadata(
        os.getenv("CPTAC_DATA_RAW_PATH")).set_index("Slide_ID")

    output = project_dir / output

    # Get device and model
    device = get_device(gpu_id)
    model, preprocess = load_model(weights_path, device)

    # Prepare HDF5 file
    hdf5_path = get_hdf5_path(weights_path, output)
    hdf5_path.parent.mkdir(exist_ok=True)

    # Store metadata in HDF5
    with h5py.File(hdf5_path, "a") as hdf5_file:
        hdf5_file.attrs["model_name"] = "resnet50"
        hdf5_file.attrs["weights_path"] = str(weights_path)
        hdf5_file.attrs["preprocessing"] = "imagenet_normalization"

    # Process each WSI
    tiles_directories = list(tiles_basedir.glob("cptac*/*/"))
    for tiles_dir in tqdm(tiles_directories):
        wsi_id = tiles_dir.stem
        tile_paths = list((tiles_dir / "tiles").glob("*.png"))

        if wsi_id not in cptac_metadata.index:
            print(f"Warning: WSI {wsi_id} not found in metadata.")
            continue

        label = cptac_metadata.loc[wsi_id, "label"]
        embeddings = compute_embeddings(
            tile_paths,
            model,
            preprocess=preprocess,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        split = "test" if wsi_id in wsi_ids_test else "train"
        store_embeddings(wsi_id, embeddings, label, hdf5_path, split)


if __name__ == "__main__":
    main()
