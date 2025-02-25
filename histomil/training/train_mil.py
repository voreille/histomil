import os

import click
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from histomil.models.mil_models import AttentionAggregatorPL
from histomil.training.utils import get_optimizer, get_scheduler, get_loss_function
from histomil.data.torch_datasets import HDF5WSIDataset  # Ensure you have this dataset class


def collate_fn_ragged(batch):
    wsi_ids, embeddings, labels = zip(*batch)
    return list(wsi_ids), list(embeddings), torch.stack(labels)


@click.command()
@click.option("--hdf5-path",
              type=str,
              required=True,
              help="Path to the HDF5 file containing embeddings.")
@click.option("--batch-size", default=128, help="Batch size for training.")
@click.option("--num-epochs", default=100, help="Number of training epochs.")
@click.option("--num-workers",
              default=4,
              help="Number of workers for DataLoader.")
@click.option("--lr", default=1e-3, help="Learning rate for the optimizer.")
@click.option("--optimizer",
              default="Adam",
              help="Optimizer to use (adam, sgd, adamw).")
@click.option("--scheduler", default="cosine", help="Learning rate scheduler.")
@click.option("--hidden-dim",
              default=128,
              help="Hidden dimension for the model.")
@click.option("--dropout", default=0.2, help="Dropout rate for the model.")
@click.option("--log-every-n", default=10, help="Logging frequency.")
@click.option("--gpu-id", default=1, help="ID of the GPU to use")
def train_aggregator(hdf5_path, batch_size, num_epochs, num_workers, lr,
                     optimizer, scheduler, hidden_dim, dropout, log_every_n,
                     gpu_id):
    """Train the Attention MIL Aggregator Model using PyTorch Lightning."""

    # Load dataset
    train_dataset = HDF5WSIDataset(hdf5_path, split="train")
    val_dataset = HDF5WSIDataset(hdf5_path, split="test")

    embedding_dim = train_dataset.embedding_dim

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_ragged,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn_ragged,
    )

    # Initialize model
    model = AttentionAggregatorPL(
        input_dim=embedding_dim,  # Assuming your embeddings are 2048-dim
        hidden_dim=hidden_dim,
        num_classes=3,  # Change if you have more classes
        dropout=dropout,
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr},
        scheduler=scheduler,
        scheduler_kwargs={"T_max": num_epochs},  # For cosine scheduler
    )

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        log_every_n_steps=log_every_n,
        precision="16-mixed",
        check_val_every_n_epoch=20,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Close datasets
    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    train_aggregator()
