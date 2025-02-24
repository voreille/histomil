import logging
import json

import pandas as pd
import h5py
import torch
import torchvision.transforms as T
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD, AdamW, RMSprop
from torch.utils.data import DataLoader, RandomSampler

from histomil.training.loss import FocalBCEWithLogitsLoss
from histomil.data.torch_datasets import (WSIDataset, EmbeddingDataset,
                                          PreloadedEmbeddingDataset,
                                          IndexedEmbeddingDataset)


def load_metadata(project_dir):
    """
    Load WSI metadata and fold information for cross-validation.
    """
    fold_df = pd.read_csv(project_dir / "data/interim/tcga_folds.csv")
    with open(project_dir / "data/interim/tcga_wsi_data.json") as f:
        wsi_metadata = json.load(f)
    return wsi_metadata, fold_df


def split_by_fold(wsi_metadata, fold_df):
    """
    Split WSI metadata by fold for k-fold cross-validation.
    """
    n_folds = fold_df["fold"].max() + 1
    output = [[] for _ in range(n_folds)]
    fold_mapping = dict(zip(fold_df['wsi_id'], fold_df['fold']))

    for wsi_info in wsi_metadata:
        fold = fold_mapping.get(wsi_info['wsi_id'])
        if fold is not None:
            output[fold].append(wsi_info)

    return output, n_folds


def get_loss_function(loss_name, **kwargs):
    loss_dict = {
        "BCEWithLogitsLoss": BCEWithLogitsLoss,
        "FocalBinaryCrossEntropy": FocalBCEWithLogitsLoss,
    }

    loss_class = loss_dict.get(loss_name)

    if loss_class is None:
        raise ValueError(f"Loss function '{loss_name}' is not supported.\n"
                         f"The available losses are: {list(loss_dict.keys())}")

    logging.info(f"Using loss function: {loss_name} with arguments: {kwargs}")

    # Check if 'weight' is in kwargs and convert it to a tensor
    if 'weight' in kwargs and not isinstance(kwargs['weight'], torch.Tensor):
        kwargs['weight'] = torch.tensor(
            kwargs['weight'],
            dtype=torch.float,
        )

    # if kwargs:
    #     return loss_class(**kwargs)
    # return loss_class()
    return loss_class(**kwargs)


def get_optimizer(parameters, optimizer_name, **kwargs):
    """
    Factory function to create an optimizer.

    Args:
        name (str): Name of the optimizer (e.g., "adam", "sgd").
        parameters: Model's parameters to optimize.
        **kwargs: Additional arguments for the optimizer.

    Returns:
        torch.optim.Optimizer: The instantiated optimizer.
    """
    optimizer_dict = {
        "Adam": Adam,
        "AdamW": AdamW,
        "SGD": SGD,
        "RMSprop": RMSprop
    }

    logging.info(f"== Optimizer: {optimizer_name} ==")

    optimizer_class = optimizer_dict.get(optimizer_name)

    if optimizer_class is None:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported")

    return optimizer_class(parameters, **kwargs)


def get_scheduler(optimizer, name, **kwargs):
    """
    Factory function to create a learning rate scheduler.

    Args:
        name (str): Name of the scheduler (e.g., "StepLR", "CosineAnnealingLR").
        optimizer: Optimizer to attach the scheduler to.
        **kwargs: Additional arguments for the scheduler.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or dict: The instantiated scheduler.
    """

    schedulers_dict = {
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

    scheduler_class = schedulers_dict.get(name)
    if scheduler_class is None:
        raise ValueError(f"Unsupported scheduler: {name}")

    if name == "ReduceLROnPlateau":
        return {
            "scheduler": scheduler_class(optimizer, **kwargs),
            "monitor": kwargs.get("monitor", "val_loss"),
            "interval": kwargs.get("interval", "epoch"),
            "frequency": kwargs.get("frequency", 1),
        }

    return scheduler_class(optimizer, **kwargs)


def get_wsi_dataloaders(wsi_metadata_by_folds, fold, label_map, batch_size=2):
    """
    Create training and validation DataLoaders based on the current fold.
    """
    wsi_meta_train = [
        wsi for i, wsi_fold in enumerate(wsi_metadata_by_folds) if i != fold
        for wsi in wsi_fold
    ]
    wsi_meta_val = wsi_metadata_by_folds[fold]

    train_dataset = WSIDataset(
        wsi_meta_train,
        label_map=label_map,
    )
    val_dataset = WSIDataset(
        wsi_meta_val,
        label_map=label_map,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader}


def collate_fn_ragged(batch):
    wsi_ids, embeddings, labels = zip(*batch)
    return list(wsi_ids), list(embeddings), torch.stack(labels)


def get_preloadedembedding_dataloaders(
    wsi_ids,
    embeddings,
    labels_numeric,
    indices,
    batch_size=1,
    shuffle=True,
    num_workers=55,
    prefetch_factor=2,
    pin_memory=False,
    resample=None,
):
    dataset = IndexedEmbeddingDataset(
        wsi_ids,
        embeddings,
        labels_numeric,
        indices=indices,
    )
    if resample:
        num_samples = 1000 * batch_size  # 1000 steps * batch size 512
        sampler = RandomSampler(dataset,
                                replacement=True,
                                num_samples=num_samples)
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_ragged,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        sampler=sampler,
    )


def get_embedding_dataloaders(
    wsi_ids,
    labels,
    hdf5_filepath,
    batch_size=1,
    preloading=True,
    shuffle=True,
    label_map=None,
    num_workers=55,
    prefetch_factor=2,
    pin_memory=False,
):

    dataset = EmbeddingDataset(
        hdf5_filepath,
        wsi_ids,
        labels,
        preloading=preloading,
        label_map=label_map,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_ragged,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )


def get_preprocessing(data_cfg):
    image_size = data_cfg["image_size"]
    mean = data_cfg["mean"]
    std = data_cfg["std"]
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
