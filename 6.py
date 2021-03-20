import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

input_size = 3 * 32 * 32
hidden_size1 = 500
hidden_size2 = 200
batch_size = 100
num_classes = 10
num_epochs = 5
learning_rate = 0.001

train_dataset = dsets.CIFAR10('/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.CIFAR10('/data', train=False, transform=transforms.ToTensor(), download=True)

train_Loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_Loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, num_classes)

    def forword(self, x):
        out = torch.relu(self.layer1(x))
        out = torch.relu(self.layer2(out))
        out = self.layer3(out)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net(input_size, hidden_size1, hidden_size2, num_classes)
net.to(device)
print(net)
print(torch.__version__)

criterion = nn.CrossEntropyLoss
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print('current epoch = %d' % epoch)
    for i, (images, labels) in enumerate(train_Loader):
        images = images.to(device)
        images = Variable(images.view(images.size(0), -1))
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('current loss = %.5f' % loss.item())

print('Finished training')
