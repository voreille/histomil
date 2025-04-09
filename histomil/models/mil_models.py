import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from histomil.training.utils import (
    get_optimizer,
    get_scheduler,
    get_loss_function,
    get_metric,
)


class AttentionAggregatorPL(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        attention_dim=128,
        hidden_dim=128,
        attention_branches=1,
        num_classes=3,
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
        self.attention_dim = attention_dim
        self.attention_branches = attention_branches
        self.num_classes = num_classes

        self.feature_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.attention_tanh = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),  # matrix V
            nn.Tanh(),
        )

        self.attention_sigmoid = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),  # matrix U
            nn.Sigmoid(),
        )

        self.attention_weights = nn.Linear(
            self.attention_dim, self.attention_branches
        )  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_dim * self.attention_branches, self.num_classes),
        )

        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.loss_fn = get_loss_function(loss, **(loss_kwargs or {}))

        # Metrics
        # self.task = "multilabel" if loss == "BCEWithLogitsLoss" else "multiclass"
        self.task = "multiclass"
        self.train_accuracy = get_metric(
            task=self.task,
            num_classes=num_classes,
            average="macro",
        )
        self.val_accuracy = get_metric(
            task=self.task,
            num_classes=num_classes,
            average="macro",
        )
        self.test_accuracy = get_metric(
            task=self.task,
            num_classes=num_classes,
            average="macro",
        )
        self.save_hyperparameters()

    def forward(self, x):
        """
        x: (num_patches, input_dim) - Each WSI has a variable number of patches
        """
        x = self.feature_projection(x)  # (num_patches, hidden_dim)

        # Compute Gated Attention Scores
        attention_tanh = self.attention_tanh(x)  # (num_patches, attention_dim)
        attention_sigmoid = self.attention_sigmoid(x)  # (num_patches, attention_dim)
        attention_scores = self.attention_weights(
            attention_tanh * attention_sigmoid
        )  # (num_patches, attention_heads)

        # Normalize attention scores
        attention_scores = torch.transpose(
            attention_scores, 1, 0
        )  # (attention_heads, num_patches)
        attention_scores = F.softmax(attention_scores, dim=1)  # Normalize over patches

        # Aggregate patch embeddings using attention
        aggregated_features = torch.mm(
            attention_scores, x
        )  # (attention_heads, hidden_dim)

        # Classification
        prediction = self.classifier(aggregated_features)  # (num_classe,)

        return prediction.squeeze(), attention_scores

    def predict_one_embedding(self, embedding):
        with torch.inference_mode():
            logit, attention_scores = self(embedding)
        pred = logit.argmax(dim=-1)
        probs = torch.sigmoid(logit)
        return pred, probs, attention_scores

    def predict_step(self, batch, batch_idx):
        wsi_ids, embeddings, labels = batch
        logits = []

        for embedding in embeddings:
            output, _ = self(embedding)
            logits.append(output)

        logits = torch.stack(logits)

        preds = logits.argmax(dim=-1)

        probs = torch.sigmoid(logits)  # Shape: (batch_size, 3)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "labels": labels,
            "wsi_ids": wsi_ids,
        }

    def step(self, batch):
        _, embeddings, labels = batch
        batch_outputs = []
        batch_size = len(labels)

        for embedding in embeddings:
            outputs, _ = self(embedding)
            batch_outputs.append(outputs)

        batch_outputs = torch.stack(batch_outputs)

        if self.loss_fn.__class__.__name__ == "BCEWithLogitsLoss":
            labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            loss = self.loss_fn(batch_outputs, labels_one_hot)
        else:
            loss = self.loss_fn(batch_outputs, labels)

        preds = batch_outputs.argmax(dim=-1)

        return loss, preds, labels, batch_size

    def training_step(self, batch, batch_idx):
        loss, preds, labels, batch_size = self.step(batch)
        self.train_accuracy(preds, labels)

        # Log metrics
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log(
            "train_acc",
            self.train_accuracy.compute(),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, preds, labels, batch_size = self.step(batch)
        self.val_accuracy(preds, labels)
        accuracy = self.val_accuracy.compute()

        # Log metrics
        self.log(
            "val_loss", val_loss, on_epoch=True, batch_size=batch_size, prog_bar=True
        )
        self.log(
            "val_acc", accuracy, on_epoch=True, batch_size=batch_size, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, preds, labels, batch_size = self.step(batch)
        self.test_accuracy(preds, labels)
        accuracy = self.test_accuracy.compute()
        # Log test metrics
        self.log("test_loss", loss, batch_size=batch_size)
        self.log("test_acc", accuracy, batch_size=batch_size, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.parameters(), self.optimizer_name, **self.optimizer_kwargs
        )
        if self.scheduler_name is None:
            return optimizer

        scheduler = get_scheduler(
            optimizer, self.scheduler_name, **self.scheduler_kwargs
        )

        if isinstance(scheduler, dict):
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return [optimizer], [scheduler]


class GatedAttention(nn.Module):
    def __init__(
        self, input_dim=2048, hidden_dim=128, attention_dim=128, attention_branches=1
    ):
        super(GatedAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attention_branches = attention_branches

        self.feature_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.attention_tanh = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),  # matrix V
            nn.Tanh(),
        )

        self.attention_sigmoid = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),  # matrix U
            nn.Sigmoid(),
        )

        self.attention_weights = nn.Linear(
            self.attention_dim, self.attention_branches
        )  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * self.attention_branches, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (num_patches, input_dim) - Each WSI has a variable number of patches
        """
        x = self.feature_projection(x)  # (num_patches, hidden_dim)

        # Compute Gated Attention Scores
        attention_tanh = self.attention_tanh(x)  # (num_patches, attention_dim)
        attention_sigmoid = self.attention_sigmoid(x)  # (num_patches, attention_dim)
        attention_scores = self.attention_weights(
            attention_tanh * attention_sigmoid
        )  # (num_patches, attention_heads)

        # Normalize attention scores
        attention_scores = torch.transpose(
            attention_scores, 1, 0
        )  # (attention_heads, num_patches)
        attention_scores = F.softmax(attention_scores, dim=1)  # Normalize over patches

        # Aggregate patch embeddings using attention
        aggregated_features = torch.mm(
            attention_scores, x
        )  # (attention_heads, hidden_dim)

        # Classification
        prediction = self.classifier(aggregated_features)  # (1,)

        return prediction, attention_scores
