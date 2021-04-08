import os
import sklearn
import numpy as np
from skimage import feature as skif
from skimage import io, transform
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
import cv2
import torch
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import joblib
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


def resize_image(filepath, filename, height_dst, width_dst, isLive='live'):
    img = cv2.imread(filepath)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    fx = width_dst / width
    fy = height_dst / height
    resize_img = cv2.resize(img, (0, 0), fx=fx, fy=fy)

    cv2.imwrite(r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize\\' + isLive + '\\' + filename, resize_img)
    return resize_img


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])  # 图像标准化处理
])
batch_size = 100
train_image_folder = torchvision.datasets.ImageFolder(r'E:\DataSets\CelebA_Spoof\New_Data\\face_region\train_datasets',
                                                      transform=transform)
train_data_loader = torch.utils.data.DataLoader(train_image_folder, batch_size=batch_size, shuffle=True)
train_hist_global = np.zeros((len(train_data_loader), batch_size, 256 * 3))
train_hist_global__ = np.zeros((len(train_data_loader), batch_size, 256 * 3, 2))
train_label_array = np.zeros((len(train_data_loader), batch_size))

test_image_folder = torchvision.datasets.ImageFolder(r'E:\DataSets\CelebA_Spoof\New_Data\\face_region\test_datasets',
                                                     transform=transform)
test_data_loader = torch.utils.data.DataLoader(test_image_folder, batch_size=batch_size, shuffle=True)
test_hist_global = np.zeros((len(test_data_loader), batch_size, 256 * 3))
test_hist_global__ = np.zeros((len(test_data_loader), batch_size, 256 * 3, 2))
test_label_array = np.zeros((len(test_data_loader), batch_size))


def rgb2ycbcr(rgb_image):
    if len(rgb_image.shape) != 3 or rgb_image.shape[0] != 3:
        raise ValueError("input image is not a rgb image")
    # rgb_image = rgb_image.astype(np.float32)
    # 1：创建变换矩阵，和偏移量
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    _, w, h = rgb_image.shape
    # 2：遍历每个像素点的三个通道进行变换
    for i in range(w):
        for j in range(h):
            ycbcr_image[:, i, j] = np.dot(transform_matrix, rgb_image[:, i, j]) + shift_matrix
    return ycbcr_image


def get_lbp_data(images_data_list, batch_id, isTrain=True, hist_size=256, lbp_radius=1, lbp_point=8):
    n_images = images_data_list.shape[0]
    hist_local = np.zeros((n_images, 3, hist_size))

    if isTrain:
        hist_global = train_hist_global
        hist_global__ = train_hist_global__
    else:
        hist_global = test_hist_global
        hist_global__ = test_hist_global__
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

        for k in range(hist_size * 3):
            hist_global__[batch_id][i][k] = [k, hist_global[batch_id][i][k]]

    return hist_global


if __name__ == '__main__':
    startTime = time.time()
    test_train1 = np.zeros((len(train_data_loader) * batch_size, 256 * 3))
    test_train2 = np.zeros((len(test_data_loader) * batch_size, 256 * 3))
    test_label1 = np.zeros((len(train_data_loader) * batch_size))
    test_label2 = np.zeros((len(test_data_loader) * batch_size))
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # ovrc = OneVsRestClassifier(svr_rbf, n_jobs=-1)
    SVC()
    sklearn.multiclass.OneVsOneClassifier()
    svr_rbf = sklearn.svm.SVR(kernel='rbf', C=2, gamma=0.1)
    classifier = OneVsRestClassifier(svr_rbf, n_jobs=-1)
    scores = []
    Sum = 0
    all = 0
    print(len(train_data_loader))
    print(len(test_data_loader))
    for batch_id, (images, labels) in enumerate(train_data_loader):
        print(batch_id / len(train_data_loader))
        train_label_array[batch_id][0:len(labels)] = labels
        print(train_label_array[batch_id])
        print(train_label_array.shape)
        print(train_label_array[batch_id].shape)

        for i in range(100):
            images[i] = (images[i] * 255)

        train_lbp_data = get_lbp_data(images, batch_id, isTrain=True)
        for k in range(100):
            test_train1[batch_id * 100 + k] = train_lbp_data[batch_id][k]
            test_label1[batch_id * 100 + k] = labels[k]
            if test_label1[batch_id * 100 + k] == 0:
                test_label1[batch_id * 100 + k] = -1

        classifier_fit = classifier.fit(train_lbp_data[batch_id], train_label_array[batch_id])

        # ovrc = joblib.load('../model/sklearn_model/ovrc.pkl')

        # joblib.dump(classifier, '../model/sklearn_model/classifier02.pkl')
        if batch_id == 30:
            classifier_fit = classifier.fit(test_train1[0:29 * batch_size], test_label1[0:29 * batch_size])
            joblib.dump(classifier, '../model/sklearn_model/classifier05.pkl')
            break
    print('结束！')
    classifier = joblib.load('../model/sklearn_model/classifier05.pkl')

    for batch_id, (images, labels) in enumerate(test_data_loader):
        print(batch_id)
        test_label_array[batch_id][0:len(labels)] = labels
        s = 0
        for i in range(100):
            images[i] = images[i] * 255
            if test_label_array[batch_id][i] == 1:
                s += 1
            else:
                test_label_array[batch_id][i] = -1
        print('s:', s)

        test_lbp_data = get_lbp_data(images, batch_id, False)
        for k in range(batch_size):
            test_train2[batch_id * 100 + k] = test_lbp_data[batch_id][k]
            test_label2[batch_id * 100 + k] = labels[k]
        print(test_train2.shape)
        # score = classifier_fit.score(test_lbp_data[batch_id], test_label_array[batch_id])
        if batch_id == 1:

            # score = classifier_fit.score(test_train2[0:14*batch_size], test_train2[0:14*batch_size])
            # scores.append(score)
            # print('分数：', score)

            sum_ = 0
            for i in range(100):
                # predict_ = classifier.predict(test_lbp_data[batch_id][i].reshape(1, -1))
                # print(test_train2[0].ravel())

                predict_ = classifier.predict(test_train2[0].reshape(1, -1))
                print(predict_)
                # print('predict:', predict_, 'test_label:', test_label_array[batch_id][i])
                if predict_ == test_label_array[batch_id][i]:
                    sum_ += 1
                    Sum += 1
                all += 1
            print(batch_id, 'acc:', sum_ / 100)
            print('allAcc:', Sum / all)
            break

    # max_s = max(scores)
    # min_s = min(scores)
    # avg_s = sum(scores) / float(16)
    # print('==========\nmax: %s\nmin: %s\navg: %s' % (max_s, min_s, avg_s))

    # sum_ = 0
    # for i in range(100):
    #     predict_ = joblib_load.predict(test_lbp_data[batch_id][i].reshape(1, -1))
    #     all += 1
    #     if predict_ == test_label_array[batch_id][0]:
    #         sum_ += 1
    #         sum += 1
    # print(batch_id, 'acc:', sum_ / 100)
    # if batch_id == len(test_data_loader) - 1:
    #     print('acc_all:', sum / all)
