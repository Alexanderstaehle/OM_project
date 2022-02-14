#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 01:21:07 2022

@author: mario
"""

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

# dropout_prob = 0.5
batch_size = 32


def load_mnist(batch_size=32, seed=42):
    torch.manual_seed(seed)

    mnist_train = dsets.MNIST(root="data/MNIST",
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)
    mnist_test = dsets.MNIST(root="data/MNIST",
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)

    test_dataloader = torch.utils.data.DataLoader(dataset=mnist_test,
                                                  shuffle=True,
                                                  batch_size=batch_size)
    train_dataloader = torch.utils.data.DataLoader(dataset=mnist_train,
                                                   shuffle=True,
                                                   batch_size=batch_size)

    return train_dataloader, test_dataloader


# train_dataloader, test_dataloader = load_mnist()
train_dataloader, test_dataloader = load_mnist(batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"


class CNN_MNIST(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(CNN_MNIST, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_prob))

        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_prob))

        self.flatten = nn.Flatten()

        self.output_layer = nn.Linear(7 * 7 * 64, 10, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        # evtl. unnötig, weil eh schon flach
        x = self.flatten(x)
        x = self.output_layer(x)
        output = self.softmax(x)

        return output


# def optimize_step(model, optimizer, criterion, learning_rate):


def optimize(train_dataloader, test_dataloader, batch_size, model, optimizer, criterion, learning_rate, epochs):
    train_cost = []
    train_acc = []

    total_batch = len(train_dataloader.dataset) // batch_size

    for epoch in range(epochs):
        avg_cost = 0

        for i, (batch_X, batch_Y) in enumerate(train_dataloader):
            X, Y = Variable(batch_X), Variable(batch_Y)

            optimizer.zero_grad()

            hypothesis = model(X)
            cost = criterion(hypothesis, Y)

            cost.backward()
            optimizer.step()

            prediction = hypothesis.data.max(dim=1)[1]
            train_acc.append(((prediction.data == Y.data).float().mean()).item())
            train_cost.append(cost.item())

            if i % 200 == 0:
                print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch + 1, i, train_cost[-1],
                                                                                          train_acc[-1]))

            avg_cost += cost.data / total_batch

        print("[Epoch: {:>4}], averaged cost = {:>.9}".format(epoch + 1, avg_cost.item()))

    print('Learning Finished!')


model = CNN_MNIST()

learning_rate = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

training_epochs = 1

optimize(train_dataloader, test_dataloader, batch_size, model, optimizer, criterion, learning_rate, training_epochs)
