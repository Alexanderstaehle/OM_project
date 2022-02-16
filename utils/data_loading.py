import random

import torch
from torchvision import datasets
from torchvision.transforms import transforms


def initialize_env(seed=42):
    """
    Sets environment variables and seeds to make model training deterministic
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_mnist(batch_size=32):
    mnist_train = datasets.MNIST(root="data/MNIST",
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True)
    mnist_test = datasets.MNIST(root="data/MNIST",
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


def load_fashion_mnist(batch_size=32):
    mnist_train = datasets.FashionMNIST(root="data/FashionMNIST",
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
    mnist_test = datasets.FashionMNIST(root="data/FashionMNIST",
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
