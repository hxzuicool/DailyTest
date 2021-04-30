import os
import sklearn
import numpy as np
from skimage import feature as skif
from skimage import io, transform
import random
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import dlib

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])
detector = dlib.get_frontal_face_detector()


# batch_size = 128
# train_image_folder = torchvision.datasets.ImageFolder(
#     r'F:\Datasets\face_region\train_datasets',
#     transform=trans)
# train_data_loader = torch.utils.data.DataLoader(train_image_folder, batch_size=batch_size, shuffle=False)
#
# test_image_folder = torchvision.datasets.ImageFolder(
#     r'F:\Datasets\face_region\test_datasets',
#     transform=trans)
# test_data_loader = torch.utils.data.DataLoader(test_image_folder, batch_size=batch_size, shuffle=False)
#
# train_image_all_num = 0
# test_image_all_num = 0
# for _, (images, labels) in enumerate(train_data_loader):
#     train_image_all_num += len(images)
# train_hist_global = np.zeros((train_image_all_num, 256 * 3))
# train_labels = np.zeros(train_image_all_num)
#
# for _, (images, labels) in enumerate(test_data_loader):
#     test_image_all_num += len(images)
# test_hist_global = np.zeros((test_image_all_num, 256 * 3))
# test_labels = np.zeros(test_image_all_num)


def get_lbp_data(images_data_list, batch_id, isTrain=True, hist_size=256, lbp_radius=1, lbp_point=8):
    n_images = images_data_list.shape[0]
    hist_local = np.zeros((n_images, 3, hist_size))
    for i in np.arange(n_images):
        image_data = images_data_list[i, :, :, :]
        img_YCbCr = cv2.cvtColor(image_data.numpy().transpose((1, 2, 0)), cv2.COLOR_RGB2HSV)
        for j in range(3):
            img = img_YCbCr[:, :, j]
            # 使用LBP方法提取图像的纹理特征.
            lbp = skif.local_binary_pattern(img, lbp_point, lbp_radius, 'default')
            # 统计图像的直方图
            max_bins = int(lbp.max() + 1)
            # hist size:256
            hist_local[i][j], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
            if isTrain:
                train_hist_global[batch_id * batch_size + i][j * 256:(j + 1) * 256] = hist_local[i][j]
            else:
                test_hist_global[batch_id * batch_size + i][j * 256:(j + 1) * 256] = hist_local[i][j]
    if isTrain:
        return train_hist_global
    else:
        return test_hist_global


def get_lbp_data_test(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = np.zeros((1, 256 * 3))
    for i in range(3):
        img = img_hsv[:, :, i]
        lbp = skif.local_binary_pattern(img, P=8, R=1)
        max_bins = int(lbp.max() + 1)
        hist[0][i * 256: (i + 1) * 256], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return hist


def detect_face(image):
    global top, bottom, left, right
    faces = detector(image, 1)

    if len(faces) == 0:
        print('没有检测到人脸。')
        return len(faces), image
    for idx, face in enumerate(faces):
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        # print(left, right, top, bottom)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        print('检测出错。')
        return len(faces), image
    Roi_img = image[top:bottom, left:right]
    return len(faces), Roi_img


if __name__ == '__main__':

    model = joblib.load('../model/sklearn_model/best_acc.pkl')
    listdir = os.listdir(r'E:\DataSets\CelebA_Spoof\New_Data\test\live')
    for i in range(100):
        face_len, img_face = detect_face(
            cv2.cvtColor(cv2.imread(r'E:\DataSets\CelebA_Spoof\New_Data\test\live\\' + listdir[i]), cv2.COLOR_BGR2RGB))
        if face_len == 0:
            continue
        lbp_data = get_lbp_data_test(img_face)
        predict = model.predict(lbp_data)
        print(predict)
