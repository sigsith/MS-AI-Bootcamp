# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.data_path, self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


# Custom CNN for image classification. (Because it's better!)
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
def load(path, batch_size, num_samples=100):
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
    indices = torch.randperm(len(dataset))[:num_samples]
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    # Load only the subset per sampler.
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return data_loader


def train(model, data_loader, n_epoch):
    model.train()
    for ep in range(n_epoch):
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = model.loss_fn(outputs, targets)
            model.backward(loss)
        # TODO: Display ROC-AUC, PR-AUC, Precision, Recall statistics.
        print(f"Epoch: {ep+1}, Loss: {loss.item()}")
    return model


if __name__ == "__main__":
    n_epoch = 20
    batch_size = 4
    num_samples = 100
    n_classes = 5
    # Warning: Can be very slow on cpu. Keep num_samples small.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = load(
        "./flower_images", batch_size=batch_size, num_samples=num_samples
    )
    model = CustomNetwork(n_classes).to(device)
    model = train(model, data_loader, n_epoch)
    model.save("trained_weights.pt")
    # TODO: Apply test samples and visualize results.
