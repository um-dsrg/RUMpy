import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicNet(nn.Module):
    """
    Basic classifer adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    def __init__(self, output_size=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.final_pool = nn.AdaptiveAvgPool2d(output_size=5)  # this modification allows the net to process any input size..
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.final_pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

