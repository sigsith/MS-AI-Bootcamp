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
from efficientnet_pytorch import EfficientNet


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.data_path, self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


# Use EfficientNet (More tuning needed).
class EfficientNetWrapper(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNetWrapper, self).__init__()
        self.effnet = EfficientNet.from_pretrained("efficientnet-b0")
        self.effnet._fc = nn.Linear(self.effnet._fc.in_features, n_classes)
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.effnet(x)
        return F.log_softmax(x, dim=1)

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Custom pure CNN.
class CustomNetwork(nn.Module):
    def __init__(self, n_classes):
        super(CustomNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.00015)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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


def load_weights(model, path_weights):
    model.load_state_dict(torch.load(path_weights))
    return model


def save(model, filename):
    torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    n_epoch = 7
    batch_size = 4
    n_classes = 5
    device = torch.device(select_backend(42))
    train_loader = load("./flower_images/training", batch_size)
    val_loader = load("./flower_images/validation", batch_size)
    model = EfficientNetWrapper(n_classes)
    model = train(
        model, train_loader, val_loader, n_epoch, n_classes, device, "results.json"
    )
    save(model, "effnet_trained_weights.pt")
