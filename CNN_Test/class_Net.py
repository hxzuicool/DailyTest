import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class Net01(nn.Module):
    def __init__(self):
        super(Net01, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 18, 3)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(18, 36, 3)

        self.dropout1 = nn.Dropout(0.5)

        self.linear1 = nn.Linear(36*108*108, 1024)

        self.dropout2 = nn.Dropout(0.1)

        self.linear2 = nn.Linear(1024, 128)
        self.linear3 = nn.Linear(128, 16)
        self.linear4 = nn.Linear(16, 2)

    def forword(self, x):

        x = self.conv1(x)
        x = self.conv2(F.relu(x))
        x = self.pool1(x)
        x = self.conv3(F.relu(x))
        x = self.dropout1(x)
        x = x.view(-1, 36 * 108 * 108)  # 相当于np.reshape()，-1表示适应层数，层数由列数决定。
        x = self.linear1(F.relu(x))
        x = self.dropout2(x)
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        x = self.linear4(F.relu(x))

        return x