import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    img = plt.imread('./images/Tom_Cruise.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(18, 15))
    plt.subplot(2, 3, 1)
    plt.imshow(img)

    plt.subplot(2, 3, 2)
    plt.imshow(img_gray)

    img_tensor = transforms.ToTensor()(img)
    img_gray_tensor = transforms.ToTensor()(img_gray)

    image = Image.open('./images/Tom_Cruise.jpg')
    plt.subplot(2, 3, 3)
    plt.imshow(image.convert('L'))

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    plt.show()
