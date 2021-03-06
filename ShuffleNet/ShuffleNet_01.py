import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

dtype = torch.FloatTensor

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip()
])

# batch_size = 125
# train_image_folder = torchvision.datasets.ImageFolder(
#     r'E:\DataSets\CelebA_Spoof\New_Data\train',
#     transform=trans)
# train_data_loader = torch.utils.data.DataLoader(train_image_folder, batch_size=batch_size, shuffle=True)


def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x


class ShuffleNetUnitA(nn.Module):
    """ShuffleNet unit for stride=1"""

    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=1,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.relu(x + out)
        return out


class ShuffleNetUnitB(nn.Module):
    """ShuffleNet unit for stride=2"""

    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitB, self).__init__()
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=2,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        out = F.relu(torch.cat([x, out], dim=1))
        return out


class ShuffleNet(nn.Module):
    """ShuffleNet for groups=3"""

    def __init__(self, groups=3, in_channels=4, num_classes=2):
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, 24, 3, stride=2, padding=1)
        stage2_seq = [ShuffleNetUnitB(24, 240, groups=3)] + \
                     [ShuffleNetUnitA(240, 240, groups=3) for i in range(3)]
        self.stage2 = nn.Sequential(*stage2_seq)
        stage3_seq = [ShuffleNetUnitB(240, 480, groups=3)] + \
                     [ShuffleNetUnitA(480, 480, groups=3) for i in range(7)]
        self.stage3 = nn.Sequential(*stage3_seq)
        stage4_seq = [ShuffleNetUnitB(480, 960, groups=3)] + \
                     [ShuffleNetUnitA(960, 960, groups=3) for i in range(3)]
        self.stage4 = nn.Sequential(*stage4_seq)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        net = self.conv1(x)
        net = F.max_pool2d(net, 3, stride=2, padding=1)
        net = self.stage2(net)
        net = self.stage3(net)
        net = self.stage4(net)
        net = F.avg_pool2d(net, 7)
        net = net.view(net.size(0), -1)
        net = self.fc(net)
        logits = F.softmax(net)
        return logits


if __name__ == "__main__":
    x = Variable(torch.randn([32, 4, 224, 224]).type(dtype),
                 requires_grad=False)

    shuffleNet = ShuffleNet()
    out = shuffleNet(x)
    print(out)
    # train_loss = 0
    # total = 0
    # correct = 0
    # lr = 0.01
    #
    # criterion = nn.CrossEntropyLoss()
    # shuffleNet = ShuffleNet()
    # shuffleNet.train()
    # optim_sgd = torch.optim.SGD(shuffleNet.parameters(), lr=lr, momentum=0.9)
    #
    # for epoch in range(10):
    #     for batch_id, (images, labels) in enumerate(train_data_loader):
    #         startTime = time.time()
    #
    #         output = shuffleNet(images)
    #         loss = criterion(output, labels)
    #
    #         optim_sgd.zero_grad()
    #         loss.backward()
    #         optim_sgd.step()
    #
    #         train_loss += loss.item()
    #         _, predicted = output.max(1)
    #         total += labels.size(0)
    #         correct += predicted.eq(labels).sum().item()
    #
    #         print(batch_id, len(train_data_loader),
    #               'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
    #                   train_loss / (batch_id + 1), 100. * correct / total, correct, total),
    #               'lr: %.4f' % lr,
    #               ' time spent: %.4f   epoch: %d' % (time.time() - startTime, epoch))
    #
    #     torch.save(model.state_dict(), r'.\ShuffleNet_local.pkl')
    # torch.save(model.state_dict(), r'.\ShuffleNet_all.pkl')
