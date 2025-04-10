from pathlib import Path

import click
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger

from histomil.models.clam_wrapper import PL_CLAM_SB
from histomil.data.torch_datasets import HDF5WSIDatasetCLAM

project_dir = Path(__file__).parents[2].resolve()

WSI_IDS_TO_DISCARD = [
    "C3L-02665-27",
    "C3L-04378-28",
    "C3N-04169-29",
    "C3N-04176-29",
    "C3N-04673-24",
    "C3N-04673-29",
]


@click.command()
@click.option(
    "--hdf5-path",
    type=str,
    required=True,
    help="Path to the HDF5 file containing embeddings.",
)
@click.option(
    "--output-name", default="checkpoint_last", help="name for the final checkpoint"
)
@click.option("--num-epochs", default=100, help="Number of training epochs.")
@click.option("--num-workers", default=4, help="Number of workers for DataLoader.")
@click.option("--lr", default=1e-4, help="Learning rate for the optimizer.")
@click.option("--dropout", default=0.25, help="Dropout rate for the model.")
@click.option("--log-every-n", default=10, help="Logging frequency.")
@click.option("--gpu-id", default=1, help="ID of the GPU to use")
def train_aggregator(
    hdf5_path,
    output_name,
    num_epochs,
    num_workers,
    lr,
    dropout,
    log_every_n,
    gpu_id,
):
    """Train the Attention MIL Aggregator Model using PyTorch Lightning."""

    # Load dataset
    with h5py.File(hdf5_path, "r") as h5_file:
        wsi_ids_train = list(h5_file["train"].keys())

    wsi_ids_train = [
        wsi_id for wsi_id in wsi_ids_train if wsi_id not in WSI_IDS_TO_DISCARD
    ]
    train_dataset = HDF5WSIDatasetCLAM(hdf5_path, wsi_ids=wsi_ids_train, split="train")
    val_dataset = HDF5WSIDatasetCLAM(hdf5_path, split="test")

    embedding_dim = train_dataset.embedding_dim

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=HDF5WSIDatasetCLAM.get_collate_fn_ragged(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=HDF5WSIDatasetCLAM.get_collate_fn_ragged(),
    )

    # Initialize model
    model = PL_CLAM_SB(
        gate=True,
        size_arg="small",
        dropout=dropout,
        k_sample=8,
        n_classes=3,
        instance_loss_fn=None,
        subtyping=False,
        embed_dim=embedding_dim,
        bag_weight=0.7,
        learning_rate=lr,
    )
    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        log_every_n_steps=log_every_n,
        precision="16-mixed",
        check_val_every_n_epoch=10,
        logger=TensorBoardLogger("tb_logs", name="CLAM_wrapper"),
    )
    trainer.logger.log_hyperparams(
        {
            "run/hdf5_path": hdf5_path,
            "run/output_name": output_name,
            "run/gpu_id": gpu_id,
            "run/num_epochs": num_epochs,
            "run/num_workers": num_workers,
            "run/log_every_n": log_every_n,
            "run/module_class": model.__class__.__name__,
        }
    )

    trainer.validate(model, val_loader)

    # Train model
    trainer.fit(model, train_loader, val_loader)

    trainer.validate(model, val_loader)
    trainer.save_checkpoint(project_dir / f"models/mil/{output_name}.ckpt")

    # Close datasets
    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    train_aggregator()
