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

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])  # 图像标准化处理
])

train_image_folder = torchvision.datasets.ImageFolder(r'E:\DataSets\CelebA_Spoof\New_Data\\face_region\train_datasets',
                                                      transform=transform)
train_data_loader = torch.utils.data.DataLoader(train_image_folder, batch_size=100, shuffle=True)
train_hist_global = np.zeros((len(train_data_loader), 100, 256 * 3))
train_label_array = np.zeros((len(train_data_loader), 100))

test_image_folder = torchvision.datasets.ImageFolder(r'E:\DataSets\CelebA_Spoof\New_Data\\face_region\test_datasets',
                                                     transform=transform)
test_data_loader = torch.utils.data.DataLoader(test_image_folder, batch_size=100, shuffle=True)
test_hist_global = np.zeros((len(test_data_loader), 100, 256 * 3))
test_label_array = np.zeros((len(test_data_loader), 100))


def get_lbp_data(images_data_list, batch_id, isTrain=True, hist_size=256, lbp_radius=1, lbp_point=8):
    n_images = images_data_list.shape[0]
    hist_local = np.zeros((n_images, 3, hist_size))
    if isTrain:
        hist_global = train_hist_global
    else:
        hist_global = test_hist_global
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

            hist_global[batch_id][i][j * 256:(j + 1) * 256] = hist_local[i][j]

    return hist_global


if __name__ == '__main__':
    clf = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
