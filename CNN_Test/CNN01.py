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


class Net01(nn.Module):
    def __init__(self):
        super(Net01, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 18, 3)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(18, 18, 3)

        self.dropout1 = nn.Dropout(0.5)

        self.linear1 = nn.Linear(18 * 108 * 108, 1024)
        self.linear2 = nn.Linear(1024, 128)
        self.linear3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(F.relu(x))
        x = self.pool1(x)
        x = self.conv3(F.relu(x))
        x = self.dropout1(x)
        x = x.view(-1, 18 * 108 * 108)  # 相当于np.reshape()，-1表示适应层数，层数由列数决定。
        x = self.linear1(F.relu(x))
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))

        return x


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)  # 输入通道3，输出通道6，卷积核3

        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入通道6，输出通道16，卷积核3

        self.pool1 = nn.MaxPool2d(2, 2)  # 卷积核2，stride为1

        self.conv3 = nn.Conv2d(16, 16, 3)

        # self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(16 * 108 * 108, 512)  # 全连接层

        self.fc2 = nn.Linear(512, 64)  # 全连接层

        self.fc3 = nn.Linear(64, 2)  # 全连接层

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        torch.nn.Dropout2d(0.1)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)

        # x = self.pool2(x)
        x = self.dropout(x)

        x = x.view(-1, 16 * 108 * 108)  # 相当于np.reshape()，-1表示适应层数，层数由列数决定。
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # 图像标准化处理
])

trainset = datasets.ImageFolder(r'F:\Datasets\face_region\train_datasets', transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=120, shuffle=True, num_workers=3)
# testset = datasets.ImageFolder(r'E:\DataSets\CelebA_Spoof\New_Data\new_test', transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=120, shuffle=True, num_workers=6)
print(len(dataloader))


# print(len(testloader))


# optimizer = optim.SGD(CNNNet.parameters(), lr=0.1, momentum=0.9)


def dataTrain():
    global optimizer
    net.train()
    train_loss = 0
    total = 0
    correct = 0
    lr = 0.02

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        startTime = time.time()
        if batch_idx % 500 == 0:
            lr /= 2
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        print(inputs.shape, targets.shape)
        optimizer.zero_grad()
        output = net(inputs)
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


def showImg():
    dataiter = iter(dataloader)
    images, labels = dataiter.__next__()
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def MobileNetV2_Train():
    MobileNetV2 = torchvision.models.MobileNetV2(num_classes=2)
    optimizer1 = optim.SGD(MobileNetV2.parameters(), 0.1, momentum=0.9)
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        output = MobileNetV2(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer1.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        print(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test():
    global best_acc
    net.eval()  # 测试模式
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            startTime = time.time()
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total),
                  ' time spent: %.4f ' % (time.time() - startTime))


if __name__ == '__main__':
    # net = torch.load('../model/CNN_face_anti_spoofing.pt')
    # net = net.to('cpu')
    criterion = nn.CrossEntropyLoss()  # 定义损失函数

    # torch.save(net, '../model/CNN_face_anti_spoofing.pt')
    # net01 = torchvision.models.VGG()
    net = Net01()
    net = net.to('cpu')
    dataTrain()
