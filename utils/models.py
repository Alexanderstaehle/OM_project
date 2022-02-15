from torch import nn
from torchvision import models


class MnistResNet(nn.Module):
    def __init__(self, in_channels=1):
        super(MnistResNet, self).__init__()

        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model = models.resnet18(pretrained=True)

        # Change the input layer to take Grayscale image, instead of RGB images.
        # Hence in_channels is set as 1 or 3 respectively
        # original definition of the first layer on the ResNet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)


class MnistCnn(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(MnistCnn, self).__init__()

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
