import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pyspng
import h5py
import numpy as np


class TileDataset(Dataset):

    def __init__(self, tile_paths, augmentation=None, preprocess=None):
        """
        Tile-level dataset that returns individual tile images from a list of paths.

        Args:
            tile_paths (list): List of paths to tile images for a WSI.
            augmentation (callable, optional): augmentation to apply to each tile image.
            transform (callable, optional): Transform to apply to each tile image.
        """
        self.tile_paths = tile_paths
        self.preprocess = preprocess
        self.augmentation = augmentation

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        with open(tile_path, 'rb') as f:
            image = pyspng.load(f.read())

        if self.augmentation:
            image = self.augmentation(image=image)['image']

        if self.preprocess:
            image = self.preprocess(image).type(torch.FloatTensor)

        return image, tile_path.stem


class HDF5WSIDataset(Dataset):
    """Dataset to load embeddings from an HDF5 file for MIL training."""

    def __init__(self, hdf5_path, split="train"):
        self.hdf5_path = hdf5_path
        self.split = split
        self.h5_file = h5py.File(hdf5_path, "r")
        self.wsi_ids = list(self.h5_file[self.split].keys())

    def __len__(self):
        return len(self.wsi_ids)

    def __getitem__(self, idx):
        """Returns (wsi_id, embeddings, label)"""
        wsi_id = self.wsi_ids[idx]
        embeddings = torch.tensor(
            self.h5_file[f"{self.split}/{wsi_id}/embeddings"][:],
            dtype=torch.float32)
        label = torch.tensor(
            self.h5_file[f"{self.split}/{wsi_id}"].attrs["label"],
            dtype=torch.long)
        return wsi_id, embeddings, label

    def close(self):
        """Ensure the HDF5 file is properly closed."""
        self.h5_file.close()
