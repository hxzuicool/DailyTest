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
    transforms.Resize((128, 128)),
    # transforms.RandomHorizontalFlip()
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

batch_size = 125
hist_size = 256
train_image_folder = torchvision.datasets.ImageFolder(
    r'E:\DataSets\3DMAD\face_images',
    transform=trans)
train_data_loader = torch.utils.data.DataLoader(train_image_folder, batch_size=batch_size, shuffle=True)

test_image_folder = torchvision.datasets.ImageFolder(
    r'E:\DataSets\3DMAD\face_images_test',
    transform=trans)
test_data_loader = torch.utils.data.DataLoader(test_image_folder, batch_size=batch_size, shuffle=True)

train_image_all_num = 0
test_image_all_num = 0
for _, (images, labels) in enumerate(train_data_loader):
    train_image_all_num += len(images)
train_hist_global = np.zeros((train_image_all_num, hist_size * 3))
train_labels = np.zeros(train_image_all_num)

for _, (images, labels) in enumerate(test_data_loader):
    test_image_all_num += len(images)
test_hist_global = np.zeros((test_image_all_num, hist_size * 3))
test_labels = np.zeros(test_image_all_num)


def get_lbp_data(images_data_list, batch_id, isTrain=True, hist_size=hist_size, lbp_radius=1, lbp_point=8):
    n_images = images_data_list.shape[0]
    hist_local = np.zeros((n_images, 3, hist_size))
    for i in np.arange(n_images):
        image_data = images_data_list[i, :, :, :]
        img_HSV = cv2.cvtColor(image_data.numpy().transpose((1, 2, 0)), cv2.COLOR_RGB2HSV)
        for j in range(3):
            img = img_HSV[:, :, j]
            # 使用LBP方法提取图像的纹理特征.
            lbp = skif.local_binary_pattern(img, lbp_point, lbp_radius, 'default')
            # 统计图像的直方图
            max_bins = int(lbp.max() + 1)
            # hist size:256
            # density为True时，返回每个区间的概率密度；为False，返回每个区间中元素的个数
            hist_local[i][j], _ = np.histogram(lbp, density=True, bins=int(max_bins), range=(0, max_bins))
            if isTrain:
                train_hist_global[batch_id * batch_size + i][j * hist_size:(j + 1) * hist_size] = hist_local[i][j]
            else:
                test_hist_global[batch_id * batch_size + i][j * hist_size:(j + 1) * hist_size] = hist_local[i][j]

        # if test_labels[batch_id * batch_size + i] == 0:
        #     plt.plot(test_hist_global[i], color='red')
        # else:
        #     plt.plot(test_hist_global[i], color='blue')
        # plt.xlim([0, len(test_hist_global[i][:])])
        # plt.axis('off')
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


def processTrainData():
    print('开始处理训练数据!', time.asctime(time.localtime(time.time())))
    for batch_id, (images, labels) in enumerate(train_data_loader):
        print(batch_id + 1, '/', len(train_data_loader))
        for i in range(len(images)):
            images[i] = images[i] * 255
            train_labels[batch_id * batch_size + i] = labels[i]

        get_lbp_data(images, batch_id, isTrain=True)
    # 数据预处理
    # train_hist_global = preprocessing.scale(train_hist_global)


def processTestData():
    print('开始处理测试数据！', time.asctime(time.localtime(time.time())))

    for batch_id, (images, labels) in enumerate(test_data_loader):
        print(batch_id + 1, '/', len(test_data_loader))
        for i in range(len(images)):
            images[i] = images[i] * 255
            test_labels[batch_id * batch_size + i] = labels[i]

        get_lbp_data(images, batch_id, isTrain=False)

    # test_hist_global = preprocessing.scale(test_hist_global)
    print('测试数据处理完成！', time.asctime(time.localtime(time.time())))


def test():
    classifier_predict = joblib.load('../model/424/424_train_best_10.1.pkl')
    # predict_value = classifier_predict.decision_function(test_hist_global)
    # print(predict_value)
    predict_ = classifier_predict.predict(test_hist_global)
    print(predict_)
    correctNum = 0
    for j in range(len(predict_)):
        if predict_[j] == test_labels[j]:
            correctNum += 1
    acc = correctNum / len(predict_)
    print('Acc: ', acc)


def train():
    params_C = np.arange(0.1, 100, 1)
    best_Acc = 0
    best_C = 0
    for Ci, C in enumerate(params_C):

        # model = SVC(C=C, gamma=gamma, cache_size=2000)
        model = make_pipeline(StandardScaler(), SVC(C=C, cache_size=2000))
        classifier = sklearn.multiclass.OneVsOneClassifier(model, n_jobs=-1)

        classifier.fit(X=train_hist_global, y=train_labels)
        joblib.dump(classifier, '../model/3DMAD/427_train_' + str(C) + '.pkl')

        predict_ = classifier.predict(test_hist_global)
        correctNum = 0
        for j in range(len(predict_)):
            if predict_[j] == test_labels[j]:
                correctNum += 1
        acc = correctNum / len(predict_)

        if acc > best_Acc:
            best_Acc = acc
            best_C = C
            joblib.dump(classifier, '../model/3DMAD/427_train_best_' + str(best_C) + '.pkl')

        print('C:{}   best_C:{}   acc:{}   best_acc:{}'.format(C, best_C, acc, best_Acc))
        print(time.asctime(time.localtime(time.time())))
    print('best_Acc:', best_Acc, '   best_C:', best_C)
    print(time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    processTrainData()
    processTestData()
    # plt.show()
    train()
    # test()
