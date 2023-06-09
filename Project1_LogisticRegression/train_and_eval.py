# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 11:19:52 2023

Please implement a 3 layer logistic regression neuron network.

The workflow of this framework should be:
Forward Function:   
    Input X, mutiply weights[0] + bias[0], Output X1
    Input X1, mutiply Weights[1] + bias[1], Output X2
    Input X2, mutiply Weights[2] + bias[2], Output X3
    Input X3, to Softmax Layer, Output Logits

Backward Function:
    Input Logits, True Label y, Output Gradient2, Update Weights[2], bias[2]
    Input Gradient2, X2, Update Weights[1], bias[1], Output Gradient1
    Input Gradient1, X1, Update Weights[0], bias[0], Output Gradient0

You can use pytorch to implement this framework.

Please split the 
"""
import torch
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


class LogisticRegression:
    def __init__(self, X_train, y_train, X_val, y_val, n_hidden):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.n_layers = len(n_hidden)
        self.n_hidden = n_hidden
        self.weights = [None for _ in range(self.n_layers)]
        self.bias = [None for _ in range(self.n_layers)]

    def forward(x):
        """
        Please Implement 3 layer forward function here.
        """
        pass

    def backward(x):
        """
        Please Implement 3 layer backward function here.
        """
        pass

    def optimizer():
        """
        Please implement MiniBatch SGD, Adam Here.
        """
        pass

    def save():
        """
        Save the model offline. Suggest to save the trained weights to .pt file.
        """
        pass

    def eval():
        """
        Please print ROC-AUC, PR-AUC, Precision, Recall.
        """
        pass


# Load image datasets and return the DataLoader
# Assumes the data is organzied by the standard directory structure.
def custom_load(path, batch_size, shuffle):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = CustomDataset(path, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


if __name__ == "__main__":
    n_epoch = 20
    n_hidden = [20, 10, 5]
    data_loader = custom_load("./flower_images", batch_size=4, shuffle=true)
    results = []
    # for ep in range(n_epoch):
    #     for X_train, y_train, X_val, y_val in data_loader():
    #         y_hat = model.forward(X_train)
    #         model.backward(y_train, y_hat)
    #         model.optimize()
    #         ROC-AUC, PR-AUC, Precision, Recall = model.eval(X_val, y_val)
    #         results.append(ROC-AUC, PR-AUC, Precision, Recall)
    # model.save()
