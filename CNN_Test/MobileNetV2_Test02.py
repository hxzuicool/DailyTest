import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class Head(nn.Module):
    def __init__(self, inp_c, out_c):
        super(Head, self).__init__()
        self.conv2d = nn.Conv2d(inp_c, out_c, 3, 2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        out = self.relu6(x)
        return out


class InvertResidual(nn.Module):
    def __init__(self, inp_c, out_c, stride, expand):
        super(InvertResidual, self).__init__()
        self.stride = stride
        out_ce = out_c * expand
        self.conv1_1 = nn.Conv2d(inp_c, out_ce, 1, bias=False)
        self.conv1_2 = nn.Conv2d(out_ce, out_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ce)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.deepwide = nn.Conv2d(out_ce, out_ce, 3, stride, padding=1, groups=out_c)
        self.relu6 = nn.ReLU6(inplace=True)
        self.shortcut = nn.Sequential()
        if stride == 1 and inp_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c))

    def forward(self, x):
        ori_x = x
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.deepwide(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.conv1_2(x)
        y = self.bn2(x)
        out = y + self.shortcut(ori_x) if self.stride == 1 else y
        return out


class MobileNetv2(nn.Module):
    def __init__(self, width_mult=1, num_classes=100):
        super(MobileNetv2, self).__init__()
        block = InvertResidual
        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]]
        input_channel = int(input_channel * width_mult)
        head_layer = Head(3, input_channel)
        self.layers = [head_layer]
        for t, c, n, s in interverted_residual_setting:
            stride = s
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.layers.append(block(input_channel, output_channel, stride, t))
                else:
                    self.layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel

        self.layers = nn.Sequential(*self.layers)

        self.conv_end = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_end = nn.BatchNorm2d(1280)
        self.relu = nn.ReLU6(inplace=True)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        x = self.relu(x)
        x = self.AdaptiveAvgPool(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


if __name__ == "__main__":
    test_input = torch.rand(1, 3, 224, 224)
    print(test_input.size())
    model = MobileNetv2()
    out = model(test_input)
    print(out.size())
