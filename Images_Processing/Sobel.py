import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d.axes3d import Axes3D



if __name__ == '__main__':
    plt.figure()
    plt.subplot(121)

    img_1 = cv2.imread('../images/Tom_Cruise_1.jpg')
    img_0 = cv2.imread('../images/Tom_Cruise_00.jpg')

    img_1_sobel = cv2.Sobel(img_1, cv2.CV_64F, 1, 0)
    img_0_sobel = cv2.Sobel(img_0, cv2.CV_64F, 1, 0)

    plt.imshow(img_1_sobel)

    plt.subplot(122)
    plt.imshow(img_0_sobel)
    plt.show()