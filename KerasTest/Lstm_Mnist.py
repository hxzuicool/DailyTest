import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import time
import numpy as np


torch.set_printoptions(threshold=np.inf)


class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_classes):
        super(Rnn, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # 此时可以从out中获得最终输出的状态h
        # x = out[:, -1, :]
        x = h_n[-1, :, :]  # -1是取最后一层lstm的输出，out和h_n是一样的。
        # print(x)
        x = self.classifier(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

net = Rnn(in_dim=28, hidden_dim=10, n_layer=2, n_classes=10)  # n_layer=2 为两层lstm堆叠，计算量增大。默认为1

net = net.to('cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        optimizer.zero_grad()
        outputs = net(torch.squeeze(inputs, 1))  # net(128, 28, 28)  输入是128批量的28*28图片. 可以有c_0和h_0，因为是第一个，所以也可以没有.
        # net(torch.squeeze(inputs, 1), (c_0, h_0))
        #  h_n是上一层输出的结果，c_n是上一层调整后的记忆
        # outputs.shape = 128, 10   输出是128批量的 10个预测值
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    print('\nEpoch: %d' % epoch)
    global best_acc
    net.eval()  # 测试模式
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            outputs = net(torch.squeeze(inputs, 1))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


if __name__ == '__main__':
    start = time.time()
    # net = torch.load('../model/lstm_minst.pt')
    for epoch in range(2):
        train(epoch)
        test(epoch)

    # torch.save(net, '../model/lstm_minst.pt')
    print(time.time() - start)
