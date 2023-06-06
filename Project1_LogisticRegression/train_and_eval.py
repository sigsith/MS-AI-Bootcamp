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


class ImpletementDataset(Dataset):
    def __init__(self, data_path):
        pass

    def __len__(self):
        pass
        # return length

    def __getitem__(self, index):
        pass
        # return X_train, y_train, X_val, y_val


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


if __name__ == "main":
    pass
    # n_epoch = 20
    # batch_size = 4
    # n_hidden = [20, 10, 5]
    # model = LogisticRegression(n_hidden)
    # data = ImpletementDataset()
    # results = []
    # data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # for ep in range(n_epoch):
    #     for X_train, y_train, X_val, y_val in data_loader():
    #         y_hat = model.forward(X_train)
    #         model.backward(y_train, y_hat)
    #         model.optimize()
    #         ROC-AUC, PR-AUC, Precision, Recall = model.eval(X_val, y_val)
    #         results.append(ROC-AUC, PR-AUC, Precision, Recall)
    # model.save()
