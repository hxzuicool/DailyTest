import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=True)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.stages = nn.Sequential(*[
            self._make_stage(3, 64, num_blocks=2, max_pooling=True),
            self._make_stage(64, 128, num_blocks=2, max_pooling=True),
            self._make_stage(128, 256, num_blocks=4, max_pooling=True),
            self._make_stage(256, 512, num_blocks=4, max_pooling=True),
            self._make_stage(512, 512, num_blocks=4, max_pooling=True)
        ])
        self.head = nn.Sequential(*[
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        ])

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, max_pooling):
        layers = [Conv(in_channels, out_channels, kernel_size=3, stride=1)]
        for _ in range(1, num_blocks):
            layers.append(Conv(out_channels, out_channels, kernel_size=3, stride=1))
        if max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.stages(x))


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # 图像标准化处理
])

trainset = datasets.ImageFolder('E:\\DataSets\\CelebA_Spoof\\New_Data\\new_train', transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=120, shuffle=True, num_workers=1)
print(len(dataloader))

if __name__ == '__main__':

    vgg = VGG19(num_classes=2).train()
    criterion = nn.CrossEntropyLoss()
    global optimizer
    train_loss = 0
    total = 0
    correct = 0
    lr = 0.02

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        startTime = time.time()
        if batch_idx % 500 == 0:
            lr /= 2
            optimizer = optim.SGD(vgg.parameters(), lr=lr, momentum=0.9)
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        optimizer.zero_grad()
        output = vgg(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(dataloader),
              'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                  train_loss / (batch_idx + 1), 100. * correct / total, correct, total),
              'lr: %.8f' % lr,
              ' time spent: %.4f ' % (time.time() - startTime))
