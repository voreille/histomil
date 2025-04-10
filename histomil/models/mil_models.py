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


class CLAMLikeAttentionAggregatorPL(pl.LightningModule):
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
        lambda_cluster=0.1,  # weight for the instance-level (clustering) loss
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attention_branches = attention_branches
        self.num_classes = num_classes
        self.lambda_cluster = lambda_cluster

        # Projection: maps raw patch features to a hidden representation.
        self.feature_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Gated attention modules (using both tanh and sigmoid).
        self.attention_tanh = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),
            nn.Tanh(),
        )
        self.attention_sigmoid = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),
            nn.Sigmoid(),
        )
        self.attention_weights = nn.Linear(self.attention_dim, self.attention_branches)

        # Bag-level classifier: uses the aggregated patch features.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_dim * self.attention_branches, self.num_classes),
        )

        # Instance-level classifier: provides patch-level predictions for clustering loss.
        self.instance_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_classes)
        )

        # Loss function for bag-level supervision.
        self.loss_fn = get_loss_function(loss, **(loss_kwargs or {}))

        # Metrics initialization (adjust task/method as needed).
        self.task = "multiclass"
        self.train_accuracy = get_metric(
            task=self.task, num_classes=num_classes, average="macro"
        )
        self.val_accuracy = get_metric(
            task=self.task, num_classes=num_classes, average="macro"
        )
        self.test_accuracy = get_metric(
            task=self.task, num_classes=num_classes, average="macro"
        )

        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        # Save hyperparameters for reproducibility.
        self.save_hyperparameters()

    def forward(self, x, bag_label=None):
        """
        Args:
            x: Tensor of shape (num_patches, input_dim) – patch features for one WSI.
            bag_label: Bag-level label (used to compute the instance-level loss during training)
                       Expected to be a scalar tensor representing the target class.
        Returns:
            bag_logits: Bag-level prediction logits (scalar or vector, depending on num_classes).
            attn_scores: Normalized attention weights of shape (attention_branches, num_patches).
            instance_logits: Patch-level (instance) predictions of shape (num_patches, num_classes).
            clustering_loss: The computed instance-level loss (0.0 if bag_label is None).
        """
        # 1. Project patch features.
        x = self.feature_projection(x)  # (num_patches, hidden_dim)

        # 2. Compute gated attention scores.
        attn_tanh = self.attention_tanh(x)  # (num_patches, attention_dim)
        attn_sigmoid = self.attention_sigmoid(x)  # (num_patches, attention_dim)
        attn_scores = self.attention_weights(
            attn_tanh * attn_sigmoid
        )  # (num_patches, attention_branches)

        # Transpose to shape: (attention_branches, num_patches)
        attn_scores = torch.transpose(attn_scores, 1, 0)
        # Normalize attention scores (softmax over patches for each branch)
        attn_scores = F.softmax(attn_scores, dim=1)

        # 3. Aggregate patch features using the attention scores.
        aggregated_features = torch.mm(
            attn_scores, x
        )  # (attention_branches, hidden_dim)
        bag_representation = aggregated_features.view(
            -1
        )  # Flatten to (hidden_dim * attention_branches)

        # 4. Bag-level prediction.
        bag_logits = self.classifier(bag_representation)

        # 5. Instance-level prediction.
        instance_logits = self.instance_classifier(x)  # (num_patches, num_classes)

        # 6. Compute clustering (instance-level) loss if bag_label is provided.
        clustering_loss = 0.0
        if bag_label is not None:
            clustering_loss = self.compute_instance_clustering_loss(
                attn_scores, instance_logits, bag_label
            )

        return bag_logits.squeeze(), attn_scores, instance_logits, clustering_loss

    def predict_one_embedding(self, embedding):
        with torch.inference_mode():
            bag_logit, attention_scores, _, _ = self(embedding)
        pred = bag_logit.argmax(dim=-1)
        probs = torch.sigmoid(bag_logit)
        return pred, probs, attention_scores

    def compute_instance_clustering_loss(self, attn_scores, instance_logits, bag_label):
        """
        Compute an instance-level loss inspired by CLAM.

        For this, we select the top 10% of patches from the first attention branch, and
        enforce that their instance-level predictions align with the bag label.

        Args:
            attn_scores: Attention weights of shape (attention_branches, num_patches).
            instance_logits: Instance-level logits of shape (num_patches, num_classes).
            bag_label: The bag-level label (scalar tensor).
        Returns:
            instance_loss: Cross-entropy loss over the top-k instance predictions.
        """
        # Work with the first attention branch.
        attn_branch = attn_scores[0]  # shape: (num_patches,)
        num_patches = attn_branch.size(0)
        k = max(1, int(0.1 * num_patches))  # Top 10% of patches

        # Get indices of the top k attended patches.
        _, topk_idx = torch.topk(attn_branch, k, largest=True)
        topk_instance_logits = instance_logits[topk_idx]  # (k, num_classes)

        # Create a tensor of bag labels repeated for each top instance.
        # (Assuming bag_label is a single integer representing the class.)
        bag_labels_for_instances = bag_label.expand(k)
        # Compute instance-level cross-entropy loss.
        instance_loss = F.cross_entropy(topk_instance_logits, bag_labels_for_instances)
        return instance_loss

    def training_step(self, batch, batch_idx):
        """
        Batch expected to contain: (wsi_ids, embeddings, bag_labels), where
        embeddings is a list/tensor of patch features for each bag.
        """
        _, embeddings, bag_labels = batch
        bag_logits_list = []
        clustering_losses = []
        batch_size = len(bag_labels)
        for embedding, bag_label in zip(embeddings, bag_labels):
            bag_logits, _, _, clust_loss = self(embedding, bag_label=bag_label)
            bag_logits_list.append(bag_logits)
            clustering_losses.append(clust_loss)

        bag_logits_all = torch.stack(bag_logits_list)
        # Bag-level loss.
        bag_loss = self.loss_fn(bag_logits_all, bag_labels)
        # Mean instance-level (clustering) loss.
        mean_clustering_loss = torch.stack(clustering_losses).mean()
        loss = bag_loss + self.lambda_cluster * mean_clustering_loss

        preds = bag_logits_all.argmax(dim=-1)
        self.train_accuracy(preds, bag_labels)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(
            "train_acc",
            self.train_accuracy.compute(),
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        _, embeddings, bag_labels = batch
        bag_logits_list = []
        batch_size = len(bag_labels)
        for embedding, bag_label in zip(embeddings, bag_labels):
            bag_logits, _, _, _ = self(embedding, bag_label=bag_label)
            bag_logits_list.append(bag_logits)

        bag_logits_all = torch.stack(bag_logits_list)
        loss = self.loss_fn(bag_logits_all, bag_labels)
        preds = bag_logits_all.argmax(dim=-1)
        self.val_accuracy(preds, bag_labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(
            "val_acc", self.val_accuracy.compute(), prog_bar=True, batch_size=batch_size
        )

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
