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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

batch_size = 128
train_image_folder = torchvision.datasets.ImageFolder(
    r'F:\Datasets\face_region\train_datasets',
    transform=trans)
train_data_loader = torch.utils.data.DataLoader(train_image_folder, batch_size=batch_size, shuffle=False)

test_image_folder = torchvision.datasets.ImageFolder(
    r'F:\Datasets\face_region\test_datasets',
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


def histogram(img):
    calc_hist = cv2.calcHist([img], [0], None, [256 * 3], [0, 256 * 3 - 1])
    print(calc_hist.shape)
    plt.plot(calc_hist)
    plt.xlim([0, 256 * 3 - 1])
    # plt.imshow(calc_hist)
    # plt.hist(img.ravel(), 255, [0, 255], color='black')
    # plt.axis('off')
    plt.show()


if __name__ == '__main__':
    params_C = np.arange(2.6, 4, 0.01)
    params_gamma = np.arange(0.001, 10, 0.1)
    best_Acc = 0
    best_C = 0
    best_gamma = 0
    print('开始处理训练数据!', time.asctime(time.localtime(time.time())))
    for batch_id, (images, labels) in enumerate(train_data_loader):
        print(batch_id + 1, '/', len(train_data_loader))
        for i in range(len(images)):
            images[i] = images[i] * 255
            train_labels[batch_id * batch_size + i] = labels[i]

        get_lbp_data(images, batch_id, isTrain=True)

    # 数据预处理
    # train_hist_global = preprocessing.scale(train_hist_global)

    print('开始处理测试数据！', time.asctime(time.localtime(time.time())))
    for batch_id, (images, labels) in enumerate(test_data_loader):
        print(batch_id + 1, '/', len(test_data_loader))
        for i in range(len(images)):
            images[i] = images[i] * 255
            test_labels[batch_id * batch_size + i] = labels[i]

        get_lbp_data(images, batch_id, isTrain=False)

    # test_hist_global = preprocessing.scale(test_hist_global)
    print('测试数据处理完成！', time.asctime(time.localtime(time.time())))

    for Ci, C in enumerate(params_C):

        # model = SVC(C=C, gamma=gamma, cache_size=2000)
        model = make_pipeline(StandardScaler(), SVC(C=C, cache_size=2000))
        classifier = sklearn.multiclass.OneVsOneClassifier(model, n_jobs=-1)
        classifier.fit(X=train_hist_global, y=train_labels)
        # joblib.dump(classifier, '../model/sklearn_model/' + 'svc' + str(p + 4) + '.pkl')
        joblib.dump(classifier, '../model/sklearn_model/svm_hsv.pkl')
        # print('模型保存！  svc' + str(p + 4), time.asctime(time.localtime(time.time())))

        # classifier = joblib.load('../model/sklearn_model/svc02.pkl')

        predict_ = classifier.predict(test_hist_global)
        correctNum = 0
        for j in range(len(predict_)):
            if predict_[j] == test_labels[j]:
                correctNum += 1
        acc = correctNum / len(predict_)

        if acc > best_Acc:
            best_Acc = acc
            best_C = C
            joblib.dump(classifier, '../model/sklearn_model/best_acc_hsv.pkl')

        print('C:{}   best_C:{}   acc:{}   best_acc:{}'.format(C, best_C, acc, best_Acc))
        print(time.asctime(time.localtime(time.time())))
        # params = classifier.get_params()  # best_C = 3.2  3.3
        # print(params)
    print('best_Acc:', best_Acc, '   best_C:', best_C)
    print(time.asctime(time.localtime(time.time())))
