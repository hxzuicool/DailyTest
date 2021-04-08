import os
import sklearn
import numpy as np
from skimage import feature as skif
from skimage import io, transform
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
import cv2
import torch
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import joblib
from sklearn import preprocessing

trans = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((256, 256))
])

batch_size = 100
train_image_folder = torchvision.datasets.ImageFolder(
    r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize\train',
    transform=trans)
train_data_loader = torch.utils.data.DataLoader(train_image_folder, batch_size=batch_size, shuffle=False)

test_image_folder = torchvision.datasets.ImageFolder(
    r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize\test',
    transform=trans)
test_data_loader = torch.utils.data.DataLoader(test_image_folder, batch_size=batch_size, shuffle=False)

train_image_all_num = 0
test_image_all_num = 0
for _, (images, labels) in enumerate(train_data_loader):
    train_image_all_num += len(images)
train_hist_global = np.zeros((train_image_all_num, 256 * 3))
train_labels = np.zeros(train_image_all_num)

for _, (images, labels) in enumerate(test_data_loader):
    test_image_all_num += len(images)
test_hist_global = np.zeros((test_image_all_num, 256 * 3))
test_labels = np.zeros(test_image_all_num)


def get_lbp_data(images_data_list, batch_id, isTrain=True, hist_size=256, lbp_radius=2, lbp_point=8):
    n_images = images_data_list.shape[0]
    hist_local = np.zeros((n_images, 3, hist_size))
    for i in np.arange(n_images):
        image_data = images_data_list[i, :, :, :]
        img_YCbCr = cv2.cvtColor(image_data.numpy().transpose((1, 2, 0)), cv2.COLOR_RGB2YCrCb)
        for j in range(3):
            img = img_YCbCr[:, :, j]
            # 使用LBP方法提取图像的纹理特征.
            lbp = skif.local_binary_pattern(img, lbp_point, lbp_radius, 'default')
            # 统计图像的直方图
            max_bins = int(lbp.max() + 1)
            # hist size:256
            hist_local[i][j], bin_edges = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
            if isTrain:
                train_hist_global[batch_id * batch_size + i][j * 256:(j + 1) * 256] = hist_local[i][j]
                return train_hist_global
            else:
                test_hist_global[batch_id * batch_size + i][j * 256:(j + 1) * 256] = hist_local[i][j]
                return test_hist_global
