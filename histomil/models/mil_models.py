import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

from histomil.training.utils import get_optimizer, get_scheduler, get_loss_function

class AttentionAggregatorPL(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_classes=2,
        dropout=0.2,
        optimizer="adam",
        optimizer_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        loss="BCEWithLogitsLoss",
        loss_kwargs=None,
    ):
        super().__init__()  # Corrected initialization

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        # Define layers
        self.projection_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=dropout),
        )
        self.pre_fc_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=0),  # FIXED: Apply softmax across patches, not classes
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.loss_fn = get_loss_function(loss, **(loss_kwargs or {}))

        # Metrics
        self.train_accuracy = Accuracy(
            num_classes=num_classes,
            average='macro',
            task="multiclass",
        )
        self.val_accuracy = Accuracy(
            num_classes=num_classes,
            average='macro',
            task="multiclass",
        )

    def forward(self, x):
        x = self.projection_layer(x)  # (num_patches, hidden_dim)
        attention = self.attention(x)  # (num_patches, num_classes)
        attention = attention.transpose(1, 0)  # (num_classes, num_patches)
        aggregated_embedding = torch.mm(attention, x)  # (num_classes, hidden_dim)
        aggregated_embedding = aggregated_embedding.view(-1, self.hidden_dim * self.num_classes)
        output = self.pre_fc_layer(aggregated_embedding)  # (1, hidden_dim)
        output = self.fc(output)  # (1, num_classes)
        return output.squeeze(), attention

    def training_step(self, batch, batch_idx):
        _, embeddings, labels = batch
        batch_outputs = []
        batch_size = len(labels)

        for embedding in embeddings:
            outputs, _ = self(embedding)
            batch_outputs.append(outputs)

        batch_outputs = torch.stack(batch_outputs)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        loss = self.loss_fn(batch_outputs, labels_one_hot)

        # Compute accuracy
        preds = torch.argmax(batch_outputs, dim=-1)
        self.train_accuracy(preds, labels)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, embeddings, labels = batch
        batch_outputs = []

        for embedding in embeddings:
            output, _ = self(embedding)
            batch_outputs.append(output)

        batch_outputs = torch.stack(batch_outputs)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        loss = self.loss_fn(batch_outputs, labels_one_hot)

        # Compute accuracy
        preds = torch.argmax(batch_outputs, dim=-1)
        self.val_accuracy(preds, labels)

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)

        return loss  # FIXED: Now returning loss for proper logging

    def test_step(self, batch, batch_idx):
        _, embeddings, labels = batch
        batch_outputs = []

        for embedding in embeddings:
            output, _ = self(embedding)
            batch_outputs.append(output)

        batch_outputs = torch.stack(batch_outputs)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        loss = self.loss_fn(batch_outputs, labels_one_hot)

        # Compute accuracy
        preds = torch.argmax(batch_outputs, dim=-1)

        # Log test metrics
        self.log("test_loss", loss)
        self.log("test_acc", Accuracy(task="multiclass", num_classes=self.num_classes)(preds, labels))

        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.optimizer_name, **self.optimizer_kwargs)
        scheduler = get_scheduler(optimizer, self.scheduler_name, **self.scheduler_kwargs)

        if isinstance(scheduler, dict):
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return [optimizer], [scheduler]
