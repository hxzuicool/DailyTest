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
import cv2


if __name__ == '__main__':
    img = cv2.imread('../images/000115.jpg_crop.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    height, width = img.shape[:2]
    fx = 512 / width
    fy = 512 / height
    resize_img = cv2.resize(img, (0, 0), fx=fx, fy=fy)
    print(resize_img.shape, fx, fy, fx*width, fy*height)
    plt.imshow(resize_img)
    plt.show()
