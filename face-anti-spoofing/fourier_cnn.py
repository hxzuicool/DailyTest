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
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip()
])

batch_size = 125
train_image_folder = torchvision.datasets.ImageFolder(
    r'F:\Datasets\whole_body\train',
    transform=trans)
train_data_loader = torch.utils.data.DataLoader(train_image_folder, batch_size=batch_size, shuffle=True)

test_image_folder = torchvision.datasets.ImageFolder(
    r'F:\Datasets\whole_body\test',
    transform=trans)
test_data_loader = torch.utils.data.DataLoader(test_image_folder, batch_size=batch_size, shuffle=True)

train_image_all_num = 0
test_image_all_num = 0
for _, (images, labels) in enumerate(train_data_loader):
    train_image_all_num += len(images)
train_hist_global = np.zeros((train_image_all_num, 200))
train_labels = np.zeros(train_image_all_num)

for _, (images, labels) in enumerate(test_data_loader):
    test_image_all_num += len(images)
test_hist_global = np.zeros((test_image_all_num, 200))
test_labels = np.zeros(test_image_all_num)


def fourier_hist(image):
    img_dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    img_fftshift = np.fft.fftshift(img_dft)
    img_fftshift_ = 20 * np.log(cv2.magnitude(img_fftshift[:, :, 0], img_fftshift[:, :, 1]))

    # cv2.normalize(img_fftshift_,img_fftshift_)
    resize = cv2.resize(img_fftshift_, (256, 256))
    print(int(resize.max() + 1))
    np_histogram, _ = np.histogram(resize, density=True, bins=int(resize.max() + 1), range=(0, int(resize.max() + 1)))

    return np_histogram[50:250]


def get_fourier_data(images_list, batch_id, isTrain=True):
    images_num = images_list.shape[0]
    for i in range(images_num):
        image_data = images_list[i, :, :, :]
        image_data = image_data.numpy().transpose((1, 2, 0))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        img_dft = cv2.dft(np.float32(image_data), flags=cv2.DFT_COMPLEX_OUTPUT)
        img_fftshift = np.fft.fftshift(img_dft)
        img_fftshift_ = 20 * np.log(cv2.magnitude(img_fftshift[:, :, 0], img_fftshift[:, :, 1]))

        np_histogram, _ = np.histogram(img_fftshift_, density=True, bins=int(img_fftshift_.max() + 1),
                                       range=(0, int(img_fftshift_.max() + 1)))

        if isTrain:
            train_hist_global[batch_id * batch_size + i] = np_histogram[50:250]
        else:
            test_hist_global[batch_id * batch_size + i] = np_histogram[50:250]


if __name__ == '__main__':
    # image_0_gray = cv2.imread('../images/Tom_Cruise.jpg', 0)
    # image_1_gray = cv2.imread('../images/real_face.jpg', 0)
    # hist0 = fourier_hist(image_0_gray)
    # plt.subplot(121)
    # plt.plot(hist0)
    #
    # hist1 = fourier_hist(image_1_gray)
    # plt.subplot(122)
    # plt.plot(hist1)
    # plt.show()
    print('开始处理训练数据!', time.asctime(time.localtime(time.time())))
    for batch_id, (images, labels) in enumerate(train_data_loader):
        print(batch_id + 1, '/', len(train_data_loader))

        for i in range(len(images)):
            images[i] = images[i] * 255
            train_labels[batch_id * batch_size + i] = labels[i]

        get_fourier_data(images, batch_id, True)

    print('开始处理测试数据！', time.asctime(time.localtime(time.time())))
    for batch_id, (images, labels) in enumerate(test_data_loader):
        print(batch_id + 1, '/', len(test_data_loader))

        for i in range(len(images)):
            images[i] = images[i] * 255
            test_labels[batch_id * batch_size + i] = labels[i]

        get_fourier_data(images, batch_id, False)

    print('测试数据处理完成！', time.asctime(time.localtime(time.time())))

    model = make_pipeline(StandardScaler(), SVC(C=10, cache_size=2000))
    classifier = sklearn.multiclass.OneVsOneClassifier(model, n_jobs=-1)
    classifier.fit(X=train_hist_global, y=train_labels)

    decision_function = classifier.decision_function(test_hist_global)
    print(decision_function)
    predict_ = classifier.predict(test_hist_global)
    correctNum = 0
    for j in range(len(predict_)):
        if predict_[j] == test_labels[j]:
            correctNum += 1
    acc = correctNum / len(predict_)
    print(acc)
