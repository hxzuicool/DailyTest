import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL.Image import Image
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import numpy as np

hasGlass_Dataset = datasets.ImageFolder('./faces', transform=transforms.ToTensor())
# noGlass_Dataset = datasets.ImageFolder('./faces', transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=hasGlass_Dataset, batch_size=100, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(12*42*52, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.softmax(self.conv1(x)))
        x = self.pool2(F.softmax(self.conv2(x)))

        x = x.view(-1, 12*42*52)
        x = F.relu(self.fc2(F.softmax(self.fc1(x))))

        return x


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


criterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    net = CNNNet()
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # dataiter = iter(train_loader)
    # images, labels = dataiter.__next__()
    # print(labels)
    # imshow(torchvision.utils.make_grid(images))
    # print(images)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            print('Finished.')

