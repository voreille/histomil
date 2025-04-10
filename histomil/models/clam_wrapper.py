import pytorch_lightning as pl
import torch
import torch.nn as nn

# Import the original CLAM models
from .model_clam import CLAM_SB, CLAM_MB


class PL_CLAM_SB(pl.LightningModule):
    """
    PyTorch Lightning module for the CLAM_SB (single branch) model.
    """

    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.25,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping: bool = False,
        embed_dim: int = 1024,
        bag_weight: float = 0.7,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        # Wrap the original CLAM_SB
        self.model = CLAM_SB(
            gate=gate,
            size_arg=size_arg,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            embed_dim=embed_dim,
        )
        # Bag-level loss (choose SVM if needed)
        self.loss_fn = nn.CrossEntropyLoss()
        self.bag_weight = bag_weight
        self.learning_rate = learning_rate
        # Optionally, save the hyperparameters.
        self.save_hyperparameters()

    def forward(self, x, label=None, instance_eval=False, attention_only=False):
        return self.model(
            x, label=label, instance_eval=instance_eval, attention_only=attention_only
        )

    def training_step(self, batch, batch_idx):
        # batch is assumed to be a tuple (data, label)
        data, label = batch
        # alwas bs = 1
        # Enable instance evaluation to get both bag and instance-level losses
        logits, Y_prob, Y_hat, _, results_dict = self.model(
            data, label=label, instance_eval=True
        )
        bag_loss = self.loss_fn(logits, label)
        # Get the instance-level loss if computed; otherwise, use 0.
        instance_loss = results_dict.get("instance_loss", 0.0)
        total_loss = self.bag_weight * bag_loss + (1 - self.bag_weight) * instance_loss

        # Compute simple accuracy for logging
        preds = Y_hat.view(-1)
        acc = (preds == label).float().mean()

        # Log losses and accuracy (these will appear in TensorBoard or the console)
        self.log("train_loss", total_loss, prog_bar=True, batch_size=1)
        self.log("train_acc", acc, prog_bar=True, batch_size=1)
        self.log("bag_loss", bag_loss, batch_size=1)
        self.log("instance_loss", instance_loss, batch_size=1)
        return total_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        # During validation we disable instance-level evaluation (or set it False)
        logits, Y_prob, Y_hat, _, _ = self.model(data, label=label, instance_eval=False)
        bag_loss = self.loss_fn(logits, label)
        preds = Y_hat.view(-1)
        acc = (preds == label).float().mean()

        self.log("val_loss", bag_loss, prog_bar=True, batch_size=1)
        self.log("val_acc", acc, prog_bar=True, batch_size=1)
        return {"val_loss": bag_loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        data, label = batch
        logits, Y_prob, Y_hat, _, _ = self.model(data, label=label, instance_eval=False)
        bag_loss = self.loss_fn(logits, label)
        preds = Y_hat.view(-1)
        acc = (preds == label).float().mean()

        self.log("test_loss", bag_loss, batch_size=1)
        self.log("test_acc", acc, batch_size=1)
        return {"test_loss": bag_loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class PL_CLAM_MB(pl.LightningModule):
    """
    PyTorch Lightning module for the CLAM_MB (multi branch) model.
    """

    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.25,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping: bool = False,
        embed_dim: int = 1024,
        bag_weight: float = 0.7,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        # Wrap the original CLAM_MB
        self.model = CLAM_MB(
            gate=gate,
            size_arg=size_arg,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            embed_dim=embed_dim,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.bag_weight = bag_weight
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x, label=None, instance_eval=False, attention_only=False):
        return self.model(
            x, label=label, instance_eval=instance_eval, attention_only=attention_only
        )

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits, Y_prob, Y_hat, _, results_dict = self.model(
            data, label=label, instance_eval=True
        )
        bag_loss = self.loss_fn(logits, label)
        instance_loss = results_dict.get("instance_loss", 0.0)
        total_loss = self.bag_weight * bag_loss + (1 - self.bag_weight) * instance_loss

        preds = Y_hat.view(-1)
        acc = (preds == label).float().mean()
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("bag_loss", bag_loss)
        self.log("instance_loss", instance_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        logits, Y_prob, Y_hat, _, _ = self.model(data, label=label, instance_eval=False)
        bag_loss = self.loss_fn(logits, label)
        preds = Y_hat.view(-1)
        acc = (preds == label).float().mean()
        self.log("val_loss", bag_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": bag_loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        data, label = batch
        logits, Y_prob, Y_hat, _, _ = self.model(data, label=label, instance_eval=False)
        bag_loss = self.loss_fn(logits, label)
        preds = Y_hat.view(-1)
        acc = (preds == label).float().mean()
        self.log("test_loss", bag_loss)
        self.log("test_acc", acc)
        return {"test_loss": bag_loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
