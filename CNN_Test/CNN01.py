import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)  # 输入通道3，输出通道6，卷积核3

        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入通道6，输出通道16，卷积核3

        self.conv3 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(32 * 218 * 218, 512)  # 全连接层

        self.fc2 = nn.Linear(512, 64)  # 全连接层

        self.fc3 = nn.Linear(64, 2)  # 全连接层

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 218 * 218)  # 相当于np.reshape()，-1表示适应层数，层数由列数决定。
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.ImageFolder('E:\\DataSets\\CelebA_Spoof\\New_Data\\test', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

CNNNet = CNNNet()
CNNNet = CNNNet.to('cpu')
criterion = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.SGD(CNNNet.parameters(), lr=0.2, momentum=0.9)


def dataTrain():
    CNNNet.train()
    train_loss = 0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        optimizer.zero_grad()
        output = CNNNet(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


if __name__ == '__main__':
    # input_data = torch.randn(1, 1, 224, 224)
    # out = CNNNet(input_data)
    # print(out)

    # loss = criterion(out, target)  # 计算损失

    # 反向传递
    # net.zero_grad()  # 清零梯度
    # loss.backward()  # 自动计算梯度、反向传递

    # optimizer.step()

    dataTrain()
