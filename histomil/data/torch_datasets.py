import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pyspng
import h5py
import numpy as np
from PIL import Image


def collate_fn_ragged(batch):
    wsi_ids, embeddings, labels = zip(*batch)
    return list(wsi_ids), list(embeddings), torch.stack(labels)


class TileDataset(Dataset):
    def __init__(self, tile_paths, preprocess=None):
        """
        Tile-level dataset that returns individual tile images from a list of paths.

        Args:
            tile_paths (list): List of paths to tile images for a WSI.
            augmentation (callable, optional): augmentation to apply to each tile image.
            transform (callable, optional): Transform to apply to each tile image.
        """
        self.tile_paths = tile_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.tile_paths)

    def _load_image(self, path):
        """Loads an image efficiently using OpenCV."""
        # img = cv2.imread(path)  # OpenCV loads as BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        with open(path, "rb") as f:
            img = pyspng.load(f.read())
        img = Image.fromarray(img).convert("RGB")  # Convert to PIL
        return img

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        image = self._load_image(tile_path)

        if self.preprocess:
            image = self.preprocess(image)

        return image, str(tile_path)


class HDF5WSIDataset(Dataset):
    """Dataset to load embeddings from an HDF5 file for MIL training."""

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            wsi_ids, embeddings, labels = zip(*batch)
            return list(wsi_ids), list(embeddings), torch.stack(labels)

        return collate_fn_ragged

    def __init__(self, hdf5_path, split="train", label_map=None):
        self.hdf5_path = hdf5_path
        self.split = split
        self.h5_file = h5py.File(hdf5_path, "r")
        self.wsi_ids = list(self.h5_file[self.split].keys())
        self.embedding_dim = self.h5_file.attrs["embedding_dim"]
        self.label_map = label_map if label_map else {"NORMAL": 0, "LUAD": 1, "LUSC": 2}

    def __len__(self):
        return len(self.wsi_ids)

    def __getitem__(self, idx):
        """Returns (wsi_id, embeddings, label)"""
        wsi_id = self.wsi_ids[idx]
        embeddings = torch.tensor(
            self.h5_file[f"{self.split}/{wsi_id}/embeddings"][:], dtype=torch.float32
        )
        label = torch.tensor(
            self.label_map[self.h5_file[f"{self.split}/{wsi_id}"].attrs["label"]],
            dtype=torch.long,
        )
        return wsi_id, embeddings, label

    def close(self):
        """Ensure the HDF5 file is properly closed."""
        self.h5_file.close()


class HDF5WSIDatasetWithTileID(Dataset):
    """Dataset to load embeddings from an HDF5 file for MIL training."""

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            wsi_ids, embeddings, labels, tile_ids = zip(*batch)
            return list(wsi_ids), list(embeddings), torch.stack(labels), list(tile_ids)

        return collate_fn_ragged

    def __init__(self, hdf5_path, split="train", label_map=None):
        self.hdf5_path = hdf5_path
        self.split = split
        self.h5_file = h5py.File(hdf5_path, "r")
        self.wsi_ids = list(self.h5_file[self.split].keys())
        self.embedding_dim = self.h5_file.attrs["embedding_dim"]
        self.label_map = label_map if label_map else {"NORMAL": 0, "LUAD": 1, "LUSC": 2}

    def __len__(self):
        return len(self.wsi_ids)

    def __getitem__(self, idx):
        """Returns (wsi_id, embeddings, label)"""
        wsi_id = self.wsi_ids[idx]
        embeddings = torch.tensor(
            self.h5_file[f"{self.split}/{wsi_id}/embeddings"][:], dtype=torch.float32
        )
        label = torch.tensor(
            self.label_map[self.h5_file[f"{self.split}/{wsi_id}"].attrs["label"]],
            dtype=torch.long,
        )

        tile_ids = self.h5_file[f"{self.split}/{wsi_id}/tile_ids"][:]

        return wsi_id, embeddings, label, tile_ids

    def close(self):
        """Ensure the HDF5 file is properly closed."""
        self.h5_file.close()


class HDF5WSIDatasetCLAM(Dataset):
    """Dataset to load embeddings from an HDF5 file for MIL training."""

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            embeddings, labels = zip(*batch)
            # Assume always bs = 1
            return embeddings[0], torch.stack(labels)

        return collate_fn_ragged

    def __init__(self, hdf5_path, wsi_ids=None, split="train", label_map=None):
        self.hdf5_path = hdf5_path
        self.split = split
        self.h5_file = h5py.File(hdf5_path, "r")
        self.wsi_ids = wsi_ids if wsi_ids else list(self.h5_file[self.split].keys())
        self.embedding_dim = self.h5_file.attrs["embedding_dim"]
        self.label_map = label_map if label_map else {"NORMAL": 0, "LUAD": 1, "LUSC": 2}

    def __len__(self):
        return len(self.wsi_ids)

    def __getitem__(self, idx):
        """Returns (wsi_id, embeddings, label)"""
        wsi_id = self.wsi_ids[idx]
        embeddings = torch.tensor(
            self.h5_file[f"{self.split}/{wsi_id}/embeddings"][:], dtype=torch.float32
        )
        label = torch.tensor(
            self.label_map[self.h5_file[f"{self.split}/{wsi_id}"].attrs["label"]],
            dtype=torch.long,
        )
        return embeddings, label

    def close(self):
        """Ensure the HDF5 file is properly closed."""
        self.h5_file.close()


class HDF5WSIDatasetCLAMWithTileID(Dataset):
    """Dataset to load embeddings from an HDF5 file for MIL training."""

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            embeddings, labels, list_tile_ids = zip(*batch)
            # Assume always bs = 1
            return embeddings[0], torch.stack(labels), list_tile_ids[0]

        return collate_fn_ragged

    def __init__(self, hdf5_path, wsi_ids=None, split="train", label_map=None):
        self.hdf5_path = hdf5_path
        self.split = split
        self.h5_file = h5py.File(hdf5_path, "r")
        self.wsi_ids = wsi_ids if wsi_ids else list(self.h5_file[self.split].keys())
        self.embedding_dim = self.h5_file.attrs["embedding_dim"]
        self.label_map = label_map if label_map else {"NORMAL": 0, "LUAD": 1, "LUSC": 2}

    def __len__(self):
        return len(self.wsi_ids)

    def __getitem__(self, idx):
        """Returns (wsi_id, embeddings, label)"""
        wsi_id = self.wsi_ids[idx]
        embeddings = torch.tensor(
            self.h5_file[f"{self.split}/{wsi_id}/embeddings"][:], dtype=torch.float32
        )
        label = torch.tensor(
            self.label_map[self.h5_file[f"{self.split}/{wsi_id}"].attrs["label"]],
            dtype=torch.long,
        )


        tile_ids = self.h5_file[f"{self.split}/{wsi_id}/tile_ids"][:]
        return embeddings, label, tile_ids

    def close(self):
        """Ensure the HDF5 file is properly closed."""
        self.h5_file.close()
