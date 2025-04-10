import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class Accuracy_Logger:
    """Accuracy logger"""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += Y_hat == Y

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


class CLAM_LightningBase(pl.LightningModule):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
        bag_weight=0.7,
        lr=1e-4,
        weight_decay=1e-5,
        optimizer="adam",
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.n_classes = n_classes
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.instance_loss_fn = instance_loss_fn
        self.bag_weight = bag_weight
        self.automatic_optimization = (
            False  # We'll manually optimize to handle combined losses
        )

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.optimizer_type} not implemented"
            )
        return optimizer

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def calculate_error(self, Y_hat, Y):
        error = 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()
        return error

    def training_step(self, batch, batch_idx):
        pass  # Implemented in subclasses

    def validation_step(self, batch, batch_idx):
        pass  # Implemented in subclasses

    def test_step(self, batch, batch_idx):
        pass  # Implemented in subclasses

    def on_train_epoch_start(self):
        self.train_acc_logger = Accuracy_Logger(n_classes=self.n_classes)
        self.train_inst_logger = Accuracy_Logger(
            n_classes=2
        )  # Binary instance classification
        self.train_loss = 0.0
        self.train_error = 0.0
        self.train_inst_loss = 0.0
        self.inst_count = 0

    def on_validation_epoch_start(self):
        self.val_acc_logger = Accuracy_Logger(n_classes=self.n_classes)
        self.val_inst_logger = Accuracy_Logger(
            n_classes=2
        )  # Binary instance classification
        self.val_loss = 0.0
        self.val_error = 0.0
        self.val_inst_loss = 0.0
        self.val_inst_count = 0
        self.val_probs = []
        self.val_labels = []

    def on_test_epoch_start(self):
        self.test_acc_logger = Accuracy_Logger(n_classes=self.n_classes)
        self.test_loss = 0.0
        self.test_error = 0.0
        self.test_probs = []
        self.test_labels = []
        self.patient_results = {}

    def calculate_auc(self, labels, probs):
        if self.n_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(
                labels, classes=[i for i in range(self.n_classes)]
            )
            for class_idx in range(self.n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(
                        binary_labels[:, class_idx], probs[:, class_idx]
                    )
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float("nan"))
            auc = np.nanmean(np.array(aucs))
        return auc, aucs

    def on_train_epoch_end(self):
        n_batches = len(self.trainer.train_dataloader)
        # Calculate average metrics
        avg_loss = self.train_loss / n_batches
        avg_error = self.train_error / n_batches
        avg_inst_loss = self.train_inst_loss / max(1, self.inst_count)

        # Log metrics
        self.log("train/loss", avg_loss)
        self.log("train/error", avg_error)
        self.log("train/acc", 1 - avg_error)
        if self.inst_count > 0:
            self.log("train/inst_loss", avg_inst_loss)

        # Log class-wise accuracy
        for i in range(self.n_classes):
            acc, correct, count = self.train_acc_logger.get_summary(i)
            if acc is not None:
                self.log(f"train/class_{i}_acc", acc)

        # Log instance-level accuracy
        if self.inst_count > 0:
            for i in range(2):  # Binary classification for instances
                acc, correct, count = self.train_inst_logger.get_summary(i)
                if acc is not None:
                    self.log(f"train/inst_class_{i}_acc", acc)

    def on_validation_epoch_end(self):
        n_batches = len(self.trainer.val_dataloaders)
        # Calculate average metrics
        avg_loss = self.val_loss / n_batches
        avg_error = self.val_error / n_batches
        avg_inst_loss = self.val_inst_loss / max(1, self.val_inst_count)

        # Calculate AUC
        if len(self.val_probs) > 0:
            val_probs_np = np.vstack(self.val_probs)
            val_labels_np = np.array(self.val_labels)
            auc, aucs = self.calculate_auc(val_labels_np, val_probs_np)
            self.log("val/auc", auc)
            if self.n_classes > 2:
                for i, class_auc in enumerate(aucs):
                    if not np.isnan(class_auc):
                        self.log(f"val/class_{i}_auc", class_auc)

        # Log basic metrics
        self.log("val/loss", avg_loss)
        self.log("val/error", avg_error)
        self.log("val/acc", 1 - avg_error)
        if self.val_inst_count > 0:
            self.log("val/inst_loss", avg_inst_loss)

        # Log class-wise accuracy
        for i in range(self.n_classes):
            acc, correct, count = self.val_acc_logger.get_summary(i)
            if acc is not None:
                self.log(f"val/class_{i}_acc", acc)

        # Log instance-level accuracy
        if self.val_inst_count > 0:
            for i in range(2):  # Binary classification for instances
                acc, correct, count = self.val_inst_logger.get_summary(i)
                if acc is not None:
                    self.log(f"val/inst_class_{i}_acc", acc)

    def on_test_epoch_end(self):
        n_batches = len(self.trainer.test_dataloaders)
        # Calculate average metrics
        avg_loss = self.test_loss / n_batches
        avg_error = self.test_error / n_batches

        # Calculate AUC
        if len(self.test_probs) > 0:
            test_probs_np = np.vstack(self.test_probs)
            test_labels_np = np.array(self.test_labels)
            auc, aucs = self.calculate_auc(test_labels_np, test_probs_np)
            self.log("test/auc", auc)
            if self.n_classes > 2:
                for i, class_auc in enumerate(aucs):
                    if not np.isnan(class_auc):
                        self.log(f"test/class_{i}_auc", class_auc)

        # Log basic metrics
        self.log("test/loss", avg_loss)
        self.log("test/error", avg_error)
        self.log("test/acc", 1 - avg_error)

        # Log class-wise accuracy
        for i in range(self.n_classes):
            acc, correct, count = self.test_acc_logger.get_summary(i)
            if acc is not None:
                self.log(f"test/class_{i}_acc", acc)

        # Save patient results as a model attribute for later access
        self.test_results = self.patient_results


class CLAM_SB_Lightning(CLAM_LightningBase):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
        bag_weight=0.7,
        lr=1e-4,
        weight_decay=1e-5,
        optimizer="adam",
    ):
        super().__init__(
            gate,
            size_arg,
            dropout,
            k_sample,
            n_classes,
            instance_loss_fn,
            subtyping,
            embed_dim,
            bag_weight,
            lr,
            weight_decay,
            optimizer,
        )

        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)

        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        # Save hyperparameters for logging
        self.save_hyperparameters()

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A, h, classifier
                        )
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}

        if return_features:
            results_dict.update({"features": M})

        return logits, Y_prob, Y_hat, A_raw, results_dict

    def training_step(self, batch, batch_idx):
        data, label = batch
        optimizer = self.optimizers()

        # Forward pass with instance evaluation
        logits, Y_prob, Y_hat, _, instance_dict = self(
            data, label=label, instance_eval=True
        )

        # Calculate loss
        loss = self.cross_entropy_loss(logits, label)
        self.train_acc_logger.log(Y_hat, label)

        # Handle instance-level loss
        instance_loss = instance_dict["instance_loss"]
        self.inst_count += 1
        self.train_inst_loss += instance_loss.item()

        # Combined loss with bag weight
        total_loss = self.bag_weight * loss + (1 - self.bag_weight) * instance_loss

        # Log instance predictions
        inst_preds = instance_dict["inst_preds"]
        inst_labels = instance_dict["inst_labels"]
        self.train_inst_logger.log_batch(inst_preds, inst_labels)

        # Update metrics
        self.train_loss += loss.item()
        error = self.calculate_error(Y_hat, label)
        self.train_error += error

        # Backward pass
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch

        # Forward pass with instance evaluation
        logits, Y_prob, Y_hat, _, instance_dict = self(
            data, label=label, instance_eval=True
        )
        self.val_acc_logger.log(Y_hat, label)

        # Calculate loss
        loss = self.cross_entropy_loss(logits, label)
        self.val_loss += loss.item()

        # Handle instance-level loss
        instance_loss = instance_dict["instance_loss"]
        self.val_inst_count += 1
        self.val_inst_loss += instance_loss.item()

        # Log instance predictions
        inst_preds = instance_dict["inst_preds"]
        inst_labels = instance_dict["inst_labels"]
        self.val_inst_logger.log_batch(inst_preds, inst_labels)

        # Store probabilities and labels for AUC calculation
        self.val_probs.append(Y_prob.cpu().numpy())
        self.val_labels.append(label.item())

        # Calculate error
        error = self.calculate_error(Y_hat, label)
        self.val_error += error

        return loss

    def test_step(self, batch, batch_idx):
        data, label = batch

        # Get slide ID if available in the dataloader
        slide_id = None
        if hasattr(self.trainer.test_dataloaders[0].dataset, "slide_data"):
            slide_id = (
                self.trainer.test_dataloaders[0]
                .dataset.slide_data["slide_id"]
                .iloc[batch_idx]
            )

        # Forward pass
        logits, Y_prob, Y_hat, _, _ = self(data)
        self.test_acc_logger.log(Y_hat, label)

        # Calculate loss
        loss = self.cross_entropy_loss(logits, label)
        self.test_loss += loss.item()

        # Store probabilities for AUC calculation
        probs = Y_prob.cpu().numpy()
        self.test_probs.append(probs)
        self.test_labels.append(label.item())

        # Store patient-level results
        if slide_id is not None:
            self.patient_results.update(
                {
                    slide_id: {
                        "slide_id": np.array(slide_id),
                        "prob": probs,
                        "label": label.item(),
                    }
                }
            )

        # Calculate error
        error = self.calculate_error(Y_hat, label)
        self.test_error += error

        return loss

    def cross_entropy_loss(self, logits, label):
        return F.cross_entropy(logits, label)


class CLAM_MB_Lightning(CLAM_LightningBase):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
        bag_weight=0.7,
        lr=1e-4,
        weight_decay=1e-5,
        optimizer="adam",
    ):
        super().__init__(
            gate,
            size_arg,
            dropout,
            k_sample,
            n_classes,
            instance_loss_fn,
            subtyping,
            embed_dim,
            bag_weight,
            lr,
            weight_decay,
            optimizer,
        )

        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=n_classes
            )
        else:
            attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=dropout, n_classes=n_classes
            )
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        # Multiple bag classifiers - one for each class
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)

        # Instance classifiers
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        # Save hyperparameters for logging
        self.save_hyperparameters()

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A[i], h, classifier
                        )
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}

        if return_features:
            results_dict.update({"features": M})

        return logits, Y_prob, Y_hat, A_raw, results_dict

    def training_step(self, batch, batch_idx):
        data, label = batch
        optimizer = self.optimizers()

        # Forward pass with instance evaluation
        logits, Y_prob, Y_hat, _, instance_dict = self(
            data, label=label, instance_eval=True
        )

        # Calculate loss
        loss = self.cross_entropy_loss(logits, label)
        self.train_acc_logger.log(Y_hat, label)

        # Handle instance-level loss
        instance_loss = instance_dict["instance_loss"]
        self.inst_count += 1
        self.train_inst_loss += instance_loss.item()

        # Combined loss with bag weight
        total_loss = self.bag_weight * loss + (1 - self.bag_weight) * instance_loss

        # Log instance predictions
        inst_preds = instance_dict["inst_preds"]
        inst_labels = instance_dict["inst_labels"]
        self.train_inst_logger.log_batch(inst_preds, inst_labels)

        # Update metrics
        self.train_loss += loss.item()
        error = self.calculate_error(Y_hat, label)
        self.train_error += error

        # Backward pass
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch

        # Forward pass with instance evaluation
        logits, Y_prob, Y_hat, _, instance_dict = self(
            data, label=label, instance_eval=True
        )
        self.val_acc_logger.log(Y_hat, label)

        # Calculate loss
        loss = self.cross_entropy_loss(logits, label)
        self.val_loss += loss.item()

        # Handle instance-level loss
        instance_loss = instance_dict["instance_loss"]
        self.val_inst_count += 1
        self.val_inst_loss += instance_loss.item()

        # Log instance predictions
        inst_preds = instance_dict["inst_preds"]
        inst_labels = instance_dict["inst_labels"]
        self.val_inst_logger.log_batch(inst_preds, inst_labels)

        # Store probabilities and labels for AUC calculation
        self.val_probs.append(Y_prob.cpu().numpy())
        self.val_labels.append(label.item())

        # Calculate error
        error = self.calculate_error(Y_hat, label)
        self.val_error += error

        return loss

    def test_step(self, batch, batch_idx):
        data, label = batch

        # Get slide ID if available in the dataloader
        slide_id = None
        if hasattr(self.trainer.test_dataloaders[0].dataset, "slide_data"):
            slide_id = (
                self.trainer.test_dataloaders[0]
                .dataset.slide_data["slide_id"]
                .iloc[batch_idx]
            )

        # Forward pass
        logits, Y_prob, Y_hat, _, _ = self(data)
        self.test_acc_logger.log(Y_hat, label)

        # Calculate loss
        loss = self.cross_entropy_loss(logits, label)
        self.test_loss += loss.item()

        # Store probabilities for AUC calculation
        # probs = Y_prob.
        ############# WAS not finiseh by claude ai
