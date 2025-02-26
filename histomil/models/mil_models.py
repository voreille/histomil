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
            nn.Softmax(
                dim=0),  # FIXED: Apply softmax across patches, not classes
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
        aggregated_embedding = torch.mm(attention,
                                        x)  # (num_classes, hidden_dim)
        aggregated_embedding = aggregated_embedding.view(
            -1, self.hidden_dim * self.num_classes)
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
        labels_one_hot = F.one_hot(labels,
                                   num_classes=self.num_classes).float()
        loss = self.loss_fn(batch_outputs, labels_one_hot)

        # Compute accuracy
        preds = torch.argmax(batch_outputs, dim=-1)
        self.train_accuracy(preds, labels)

        # Log metrics
        self.log("train_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size,
                 prog_bar=True)
        self.log("train_acc",
                 self.train_accuracy,
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, embeddings, labels = batch
        batch_outputs = []

        for embedding in embeddings:
            output, _ = self(embedding)
            batch_outputs.append(output)

        batch_outputs = torch.stack(batch_outputs)
        labels_one_hot = F.one_hot(labels,
                                   num_classes=self.num_classes).float()
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
        labels_one_hot = F.one_hot(labels,
                                   num_classes=self.num_classes).float()
        loss = self.loss_fn(batch_outputs, labels_one_hot)

        # Compute accuracy
        preds = torch.argmax(batch_outputs, dim=-1)

        # Log test metrics
        self.log("test_loss", loss)
        self.log(
            "test_acc",
            Accuracy(task="multiclass", num_classes=self.num_classes)(preds,
                                                                      labels))

        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.optimizer_name,
                                  **self.optimizer_kwargs)
        scheduler = get_scheduler(optimizer, self.scheduler_name,
                                  **self.scheduler_kwargs)

        if isinstance(scheduler, dict):
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return [optimizer], [scheduler]


class GatedAttention(nn.Module):

    def __init__(self,
                 feature_dim=2048,
                 hidden_dim=128,
                 attention_dim=128,
                 attention_branches=1):
        super(GatedAttention, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attention_branches = attention_branches

        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.attention_tanh = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),  # matrix V
            nn.Tanh())

        self.attention_sigmoid = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),  # matrix U
            nn.Sigmoid())

        self.attention_weights = nn.Linear(
            self.attention_dim, self.attention_branches
        )  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * self.attention_branches, 1),
            nn.Sigmoid())

    def forward(self, x):

        x = self.feature_projection(x)  # KxM

        attention_tanh = self.attention_tanh(x)  # KxL
        attention_sigmoid = self.attention_sigmoid(x)  # KxL
        attention_scores = self.attention_weights(
            attention_tanh * attention_sigmoid
        )  # element wise multiplication # KxATTENTION_BRANCHES
        attention_scores = torch.transpose(attention_scores, 1,
                                           0)  # ATTENTION_BRANCHESxK
        attention_scores = F.softmax(attention_scores, dim=1)  # softmax over K

        aggregated_features = torch.mm(attention_scores,
                                       x)  # ATTENTION_BRANCHESxM

        prediction = self.classifier(aggregated_features)

        return prediction, attention_scores


class MyGatedAttention(nn.Module):

    def __init__(self,
                 input_channels=1,
                 feature_dim=500,
                 attention_dim=128,
                 num_classes=1,
                 attention_heads=1):
        """
        Gated Attention MIL Model.

        Args:
            input_channels (int): Number of input channels in the CNN.
            feature_dim (int): Feature space dimensionality after CNN (default: 500).
            attention_dim (int): Hidden dimension for attention computation (default: 128).
            num_classes (int): Number of output classes (default: 1, binary classification).
            attention_heads (int): Number of attention heads (default: 1).
        """
        super().__init__()

        self.feature_dim = feature_dim  # M (Feature space size after CNN)
        self.attention_dim = attention_dim  # L (Hidden dimension for attention mechanism)
        self.attention_heads = attention_heads  # ATTENTION_BRANCHES

        # CNN Feature Extractor
        self.feature_extractor_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 20, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(2, stride=2), nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(), nn.MaxPool2d(2, stride=2))

        # Fully Connected Feature Projection
        self.feature_projection = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.feature_dim),
            nn.ReLU(),
        )

        # Gated Attention Mechanism
        self.attention_tanh = nn.Sequential(
            nn.Linear(self.feature_dim, self.attention_dim),  # Matrix U
            nn.Tanh())

        self.attention_sigmoid = nn.Sequential(
            nn.Linear(self.feature_dim, self.attention_dim),  # Matrix V
            nn.Sigmoid())

        self.attention_weights = nn.Linear(self.attention_dim,
                                           self.attention_heads)  # Matrix w

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * self.attention_heads, num_classes),
            nn.Sigmoid() if num_classes == 1 else
            nn.Identity()  # Use Sigmoid for binary classification
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, channels, height, width).
        
        Returns:
            tuple: (Prediction, Attention Scores)
        """
        x = x.squeeze(0)  # Remove batch dimension if needed

        # Extract features from CNN
        features = self.feature_extractor_cnn(x)  # (num_patches, 50, 4, 4)
        features = features.view(-1, 50 * 4 *
                                 4)  # Flatten (num_patches, feature_dim)

        # Project features into feature space
        features = self.feature_projection(
            features)  # (num_patches, feature_dim)

        # Compute Gated Attention
        attention_tanh = self.attention_tanh(
            features)  # (num_patches, attention_dim)
        attention_sigmoid = self.attention_sigmoid(
            features)  # (num_patches, attention_dim)
        attention_scores = self.attention_weights(
            attention_tanh *
            attention_sigmoid)  # (num_patches, attention_heads)

        # Normalize attention scores
        attention_scores = torch.transpose(attention_scores, 1,
                                           0)  # (attention_heads, num_patches)
        attention_scores = F.softmax(attention_scores,
                                     dim=1)  # Normalize over patches

        # Compute WSI embedding
        aggregated_features = torch.mm(
            attention_scores, features)  # (attention_heads, feature_dim)

        # Classification
        prediction = self.classifier(aggregated_features)  # (num_classes,)

        return prediction, attention_scores
