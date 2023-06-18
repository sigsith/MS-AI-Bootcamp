# Basic helper library for training and evaluation. Mostly boilerplate code.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import json
from tabulate import tabulate
import random


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.data_path, self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


# Load image resources and return the DataLoader.
# Assume the standard directory structure.
def load(path, batch_size):
    # (3, 224, 224) 3D tensors.
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = CustomDataset(path, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def train_one_epoch(model, train_loader, device):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = model.loss_fn(outputs, targets)
        model.backward(loss)
    return model


def evaluate(model, val_loader, device):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_outputs.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    y_pred = np.array(all_outputs)
    y_true = np.array(all_labels)
    return y_true, y_pred


def compute_metrics(y_true, y_pred, n_classes):
    # Prepare multilabel format for multi-class data.
    lb = LabelBinarizer()
    lb.fit(range(n_classes))
    y_true_bin = lb.transform(y_true)
    y_pred_bin = lb.transform(y_pred)
    # Calculate metrics for each class.
    roc_aucs = roc_auc_score(y_true_bin, y_pred_bin, average=None)
    pr_aucs = average_precision_score(y_true_bin, y_pred_bin, average=None)
    precisions = precision_score(y_true_bin, y_pred_bin, average=None)
    recalls = recall_score(y_true_bin, y_pred_bin, average=None)
    metrics = {
        "roc_auc": roc_aucs,
        "pr_auc": pr_aucs,
        "precision": precisions,
        "recall": recalls,
    }
    return metrics


def train(model, train_loader, val_loader, n_epoch, n_classes, device, result_file):
    model.to(device)
    for ep in range(n_epoch):
        print(f"Epoch: {ep+1}")
        model = train_one_epoch(model, train_loader, device)
        y_true, y_pred = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred, n_classes)
        with open(result_file, "a") as f:
            json.dump({"epoch": ep + 1, "metrics": metrics}, f, default=list)
            f.write("\n")
        print_metrics(metrics, n_classes)
    return model


def print_metrics(metrics, n_classes):
    headers = ["Class", "ROC-AUC", "PR-AUC", "Precision", "Recall"]
    table_data = []
    for i in range(n_classes):
        row = [
            f"Class {i+1}",
            metrics["roc_auc"][i],
            metrics["pr_auc"][i],
            metrics["precision"][i],
            metrics["recall"][i],
        ]
        table_data.append(row)
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


def select_backend(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        return "cuda"
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        return "mps"
    return "cpu"


def pick_device(seed=random.randint(0, 0xFFFF_FFFF)):
    return torch.device(select_backend(seed))


def load_weights(model, path_weights):
    model.load_state_dict(torch.load(path_weights))
    return model


def save(model, filename):
    torch.save(model.state_dict(), filename)
