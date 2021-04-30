import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import torchvision
import cv2
from PIL import Image
import numpy as np
import dlib
from skimage.feature import local_binary_pattern
import h5py
from skimage import feature as skif


def t1():
    for C in range(10):
        print('sss' + str(C) + 'sss')


def t2():
    fileName_list = os.listdir(r'F:\Datasets\WMCA_RGB\preprocessed-face-station_RGB\01.02.18')
    print(fileName_list)
    index = 1
    for i in range(len(fileName_list)):
        file = h5py.File(r'F:\Datasets\WMCA_RGB\preprocessed-face-station_RGB\01.02.18\\' + fileName_list[i], 'r')

        # print(file.filename, ":")
        # print([key for key in file.values()], "\n")
        f = file['Frame_0']
        print([key for key in f.keys()])
        print(f['array'][:].transpose((1, 2, 0)))
        # for j in range(len(file['Color_Data'][:])):
        #     cv2.imwrite(r'E:\DataSets\3DMAD\images03\\' + str(index) + '.jpg',
        #                 cv2.cvtColor(file['Color_Data'][j].transpose((1, 2, 0)), cv2.COLOR_BGR2RGB))
        #     index = index + 1

        plt.imshow(f['array'][:].transpose((1, 2, 0)))

        plt.show()


def get_face_region(imagePath):
    globalImagePath = imagePath
    global top, bottom, left, right
    img = cv2.imread(globalImagePath)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(RGB_img, 1)
    if len(faces) == 0:
        return list
    for idx, face in enumerate(faces):
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        print(left, right, top, bottom)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return list
    Roi_img = RGB_img[top:bottom, left:right]

    face_img = cv2.resize(Roi_img, (224, 224))
    return face_img


def get_face_lbp(image, isSpoof):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    for i in range(3):
        img = image_hsv[:, :, i]
        lbp = skif.local_binary_pattern(img, 8, 1, 'default')
        max_bins = int(lbp.max() + 1)
        # hist = np.zeros((max_bins, 1))
        histogram, _ = np.histogram(lbp, density=True, bins=int(max_bins - 6), range=(0, max_bins - 6))
        if isSpoof:
            plt.plot(histogram, color='black')
            plt.xlim([0, len(histogram)])
            plt.axis('off')
        else:
            plt.plot(histogram, color='red')
            plt.xlim([0, len(histogram)])
            plt.axis('off')


if __name__ == '__main__':
    list = []
    spoofList = os.listdir(r'F:\Datasets\images\test_error\spoof')
    liveList = os.listdir(r'F:\Datasets\images\test_error\live')

    detector = dlib.get_frontal_face_detector()
    for i in range(len(spoofList)):

        face_region = get_face_region(r'F:\Datasets\images\test_error\spoof\\' + spoofList[i])
        if len(face_region) != 0:
            get_face_lbp(face_region, True)

    for i in range(len(liveList)):
        face_region = get_face_region(r'F:\Datasets\images\test_error\live\\' + liveList[i])
        if len(face_region) != 0:
            get_face_lbp(face_region, False)

    plt.show()
    # bgr = cv2.imread('../images/env/000897.jpg')
    #
    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #
    # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # r, g, b = cv2.split(rgb)
    #
    # face_bgr = cv2.imread('../images/env/000897_face.jpg')
    #
    # face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    #
    # face_r, face_g, face_b = cv2.split(face_rgb)
    #
    # face_hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    #
    # face_h, face_s, face_v = cv2.split(face_hsv)
    #
    # lbp = local_binary_pattern(face_v, 8, 1)
    # histogram_all(lbp)

    # plt.imshow(face_v, cmap='gray')
    # plt.show()

    # cv2.imwrite('../images/env/000897_face_v.jpg', face_v)
