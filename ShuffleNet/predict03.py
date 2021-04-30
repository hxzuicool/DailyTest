import paddlex as pdx
import cv2
import os
import shutil
import dlib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np
import joblib
import torch
import torchvision
from skimage import feature as skif

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224))
])


def predict03_1(path, threshold):
    index = 0
    RightNums_CNN = 0
    rightNums_lbp = 0
    rightNums = 0
    FA = 0
    FR = 0
    TP = 0
    live_samples_nums = 0
    spoof_samples_nums = 0
    total_samples = 0
    hist = np.zeros((256 * 3))
    lbp_live_error = []
    for live_or_spoof in ['live']:

        folderPath = path + '\\' + live_or_spoof
        imagesName_list = os.listdir(folderPath)
        if live_or_spoof == 'live':
            live_samples_nums = len(imagesName_list)
        else:
            spoof_samples_nums = len(imagesName_list)
        total_samples = live_samples_nums + spoof_samples_nums

        for idx in range(len(imagesName_list)):
            predict = ''
            filePath = folderPath + '\\' + imagesName_list[idx]

            img = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2RGB)
            faces = detector(img, 1)

            if len(faces) == 0:
                index = index + 1
                continue
            left = faces[0].left()
            right = faces[0].right()
            top = faces[0].top()
            bottom = faces[0].bottom()

            if left < 0 or right < 0 or top < 0 or bottom < 0:
                index = index + 1
                continue
            img_face = img[top:bottom, left:right]

            img_face_torch = trans(img_face) * 255
            img_face_hsv = cv2.cvtColor(img_face_torch.numpy().transpose((1, 2, 0)), cv2.COLOR_RGB2HSV)

            for j, face_img_channel in enumerate(cv2.split(img_face_hsv)):
                lbp = local_binary_pattern(face_img_channel, 8, 1)
                max_bins = int(lbp.max() + 1)
                histogram, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
                hist[256 * j:256 * (j + 1)] = histogram

            predict_lbp = classifier_predict.predict(hist.reshape(1, -1))
            predict_lbp_value = classifier_predict.decision_function(hist.reshape(1, -1))
            index = index + 1
            print(index)
            if predict_lbp == 1:
                lbp_live_error.append(predict_lbp_value)

    print(lbp_live_error)
    print(get_median(lbp_live_error))


def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:  # 判断列表长度为偶数
        median = (data[size // 2] + data[size // 2 - 1]) / 2
        data[0] = median
    if size % 2 == 1:  # 判断列表长度为奇数
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]


if __name__ == '__main__':
    print("Loading model...")
    classifier_predict = joblib.load('../model/424/424_train_best_10.1.pkl')
    model = pdx.deploy.Predictor(r'E:\Report\CelebA-Spoof\P0001-T0001_export_model\inference_model', use_gpu=False)
    detector = dlib.get_frontal_face_detector()
    print("Model loaded.")

    predict03_1(r'F:\Datasets\images\test_sub', threshold=0.5)
