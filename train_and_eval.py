# -*- coding: utf-8 -*-
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


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.data_path, self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


# Custom CNN for image classification.
class CustomNetwork(nn.Module):
    def __init__(self, n_classes):
        super(CustomNetwork, self).__init__()
        # Two convolutional layers followed by two fully connected layers.

        # Convolutional Layer 1:
        # Input: 3 channels (RGB), Output: 64 feature maps, Kernel size: 3x3.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Convolutional Layer 2:
        # Input: 64 feature maps, Output: 128 feature maps, Kernel size: 3x3.
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layer 1:
        # Input: 128x56x56 features (after two max-pooling layers),
        # Output: 512 features.
        self.fc1 = nn.Linear(128 * 56 * 56, 512)

        # Fully Connected Layer 2:
        # Input: 512 features, Output: n_classes.
        self.fc2 = nn.Linear(512, n_classes)

        # Stochastic Gradient Descent optimizer with learning rate of 0.01.
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

        # Cross-Entropy Loss Function for multi-class classification.
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Convolutional Layer 1 with rectifier and max pooling.
        # Max pooling is set to kernel size 2x2 and stride 2.
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Convolutional Layer 2. Same operation.
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Flatten into a 2D tensor.
        # Tensor size: [batch_size, 128*56*56].
        x = x.view(x.size(0), -1)

        # Fully Connected Layer 1 followed by rectifier.
        x = F.relu(self.fc1(x))

        # Fully Connected Layer 2 (output layer). No relu.
        x = self.fc2(x)

        # Log-softmax activation to obtain class probabilities.
        return F.log_softmax(x, dim=1)

    def backward(self, loss):
        # Reset gradients.
        self.optimizer.zero_grad()

        # Compute gradients of the loss.
        loss.backward()

        # Update parameters.
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


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
    # Calculate average metrics.
    roc_auc_avg = roc_aucs.mean()
    pr_auc_avg = pr_aucs.mean()
    precision_avg = precisions.mean()
    recall_avg = recalls.mean()
    metrics = {
        "roc_auc": roc_aucs,
        "pr_auc": pr_aucs,
        "precision": precisions,
        "recall": recalls,
        "roc_auc_avg": roc_auc_avg,
        "pr_auc_avg": pr_auc_avg,
        "precision_avg": precision_avg,
        "recall_avg": recall_avg,
    }
    return metrics


def train(model, train_loader, val_loader, n_epoch, n_classes, device):
    model.to(device)
    for ep in range(n_epoch):
        model = train_one_epoch(model, train_loader, device)
        y_true, y_pred = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred, n_classes)
        print(f"Epoch: {ep+1}")
        for i in range(n_classes):
            print(f"Class {i}:")
            print(f"ROC-AUC: {metrics['roc_auc'][i]}")
            print(f"PR-AUC: {metrics['pr_auc'][i]}")
            print(f"Precision: {metrics['precision'][i]}")
            print(f"Recall: {metrics['recall'][i]}")
        print("Average metrics:")
        print(f"ROC-AUC: {metrics['roc_auc_avg']}")
        print(f"PR-AUC: {metrics['pr_auc_avg']}")
        print(f"Precision: {metrics['precision_avg']}")
        print(f"Recall: {metrics['recall_avg']}")
    return model


def select_backend(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        return "cuda"
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        return "mps"
    return "cpu"


if __name__ == "__main__":
    n_epoch = 10
    batch_size = 4
    n_classes = 5
    device = torch.device(select_backend(42))
    train_loader = load("./flower_images/training", batch_size)
    val_loader = load("./flower_images/validation", batch_size)
    model = CustomNetwork(n_classes)
    model = train(model, train_loader, val_loader, n_epoch, n_classes, device)
    model.save("trained_weights.pt")
