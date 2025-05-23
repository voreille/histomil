import logging

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, SGD, AdamW, RMSprop
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy

from histomil.training.loss import FocalBCEWithLogitsLoss


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


def get_loss_function(loss_name, **kwargs):
    loss_dict = {
        "BCEWithLogitsLoss": BCEWithLogitsLoss,
        "FocalBinaryCrossEntropyLoss": FocalBCEWithLogitsLoss,
        "CrossEntropyLoss": CrossEntropyLoss,
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


def get_metric(
    num_classes,
    threshold=0.5,
    task="multilabel",
    average="macro",
    multidim_average="global",
    top_k=1,
):
    if task == "multilabel":
        return MultilabelAccuracy(
            num_labels=num_classes,
            threshold=threshold,
            average=average,
            multidim_average=multidim_average,
        )

    return MulticlassAccuracy(
        num_classes=num_classes,
        average=average,
        multidim_average=multidim_average,
        top_k=top_k,
    )


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
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
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


def collate_fn_ragged(batch):
    wsi_ids, embeddings, labels = zip(*batch)
    return list(wsi_ids), list(embeddings), torch.stack(labels)
