import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d.axes3d import Axes3D


def Display_color_difference(img_fake, img_real):
    plt.figure(figsize=(20, 15))

    plt.subplot(2, 3, 1)
    plt.imshow(img_real)
    plt.title("Real", fontsize='xx-large', fontweight='bold')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img_real, cv2.COLOR_BGR2HSV))
    plt.title("HSV", fontsize='xx-large', fontweight='bold')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(img_real, cv2.COLOR_BGR2YCrCb))
    plt.title("YCrCb", fontsize='xx-large', fontweight='bold')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    plt.subplot(2, 3, 4)
    plt.imshow(img_fake)
    plt.title("Fake", fontsize='xx-large', fontweight='bold')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(img_fake, cv2.COLOR_BGR2HSV))
    plt.title("HSV", fontsize='xx-large', fontweight='bold')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(img_fake, cv2.COLOR_BGR2YCrCb))
    plt.title("YCrCb", fontsize='xx-large', fontweight='bold')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.show()


def imgTo3D(img, cmap='gist_rainbow'):  # 也可为'hot'
    fig = plt.figure(figsize=(15, 10))
    axes3d = Axes3D(fig)
    Y = np.arange(0, np.shape(img)[0], 1)
    X = np.arange(0, np.shape(img)[1], 1)
    X, Y = np.meshgrid(X, Y)
    axes3d.plot_surface(X, Y, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap=cmap)
    plt.show()


if __name__ == '__main__':
    img1 = plt.imread('./images/Tom_Cruise_1.jpg')
    img0 = plt.imread('./images/Tom_Cruise_00.jpg')
    Display_color_difference(img0, img1)

    # imgTo3D(img1)
    # plt.figure()
    # img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img0_gray, cmap='gray')
    # plt.show()
