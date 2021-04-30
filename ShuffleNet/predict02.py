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


def predict_CelebA_Spoof(path, threshold):
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
    lbp_1 = []
    lbp_0 = []
    for live_or_spoof in ['live', 'spoof']:

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

            predict_shuffleNet_ = model.predict(filePath)
            if predict_shuffleNet_[0]['score'] > threshold and predict_shuffleNet_[0]['category'] == 'live':
                predict_shuffleNet = 'live'
            else:
                predict_shuffleNet = 'spoof'

            if predict_shuffleNet_[0]['score'] > threshold and predict_shuffleNet_[0]['category'] == 'spoof':
                predict_shuffleNet = 'spoof'
            else:
                predict_shuffleNet = 'live'

            img = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2RGB)
            faces = detector(img, 1)

            if len(faces) == 0:
                if predict_shuffleNet == 'spoof':
                    predict = 'spoof'
                else:
                    predict = 'live'
                # predict = 'spoof'
                index = index + 1

                if predict == live_or_spoof:
                    rightNums = rightNums + 1
                elif predict == 'live' and live_or_spoof == 'spoof':
                    FA = FA + 1
                elif predict == 'spoof' and live_or_spoof == 'live':
                    FR = FR + 1

                if predict == live_or_spoof == 'live':
                    TP = TP + 1
                continue
            left = faces[0].left()
            right = faces[0].right()
            top = faces[0].top()
            bottom = faces[0].bottom()

            if left < 0 or right < 0 or top < 0 or bottom < 0:
                if predict_shuffleNet == 'spoof':
                    predict = 'spoof'
                else:
                    predict = 'live'
                index = index + 1

                if predict == live_or_spoof:
                    rightNums = rightNums + 1
                elif predict == 'live' and live_or_spoof == 'spoof':
                    FA = FA + 1
                elif predict == 'spoof' and live_or_spoof == 'live':
                    FR = FR + 1

                if predict == live_or_spoof == 'live':
                    TP = TP + 1
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
            # predict_lbp_value = classifier_predict.decision_function(hist.reshape(1, -1))
            if predict_shuffleNet == 'live' and predict_lbp == 0:
                predict = 'live'
            elif predict_shuffleNet == 'spoof' and predict_lbp == 1:
                predict = 'spoof'
            elif predict_shuffleNet == 'live' and predict_lbp == 1 and classifier_predict.decision_function(
                    hist.reshape(1, -1)) > 1.23:
                predict = 'spoof'
            elif predict_shuffleNet == 'live' and predict_lbp == 1 and classifier_predict.decision_function(
                    hist.reshape(1, -1)) < 1.23:
                predict = 'live'
            elif predict_shuffleNet == 'spoof' and predict_lbp == 0 and classifier_predict.decision_function(
                    hist.reshape(1, -1)) < -0.227:
                predict = 'live'
            elif predict_shuffleNet == 'spoof' and predict_lbp == 0 and classifier_predict.decision_function(
                    hist.reshape(1, -1)) > -0.227:
                predict = 'spoof'

            if predict == live_or_spoof:
                rightNums = rightNums + 1
            elif predict == 'live' and live_or_spoof == 'spoof':
                # print(classifier_predict.decision_function(hist.reshape(1, -1)))
                FA = FA + 1
            elif predict == 'spoof' and live_or_spoof == 'live':
                # lbp_live_error.append(classifier_predict.decision_function(hist.reshape(1, -1)))
                # print(classifier_predict.decision_function(hist.reshape(1, -1)))
                FR = FR + 1
            if predict == live_or_spoof == 'live':
                TP = TP + 1
            index = index + 1
            # print(index, '/', total_samples, predict_lbp, predict_shuffleNet, predict,
            #       'acc:{:.5f}'.format(rightNums / index),
            #       'rightNums:', rightNums, 'FA:', FA, 'FR:', FR)
            if predict_lbp == 1:
                lbp_1.append(classifier_predict.decision_function(hist.reshape(1, -1)))
            elif predict_lbp == 0:
                lbp_0.append(classifier_predict.decision_function(hist.reshape(1, -1)))
    lbp_1.sort()
    lbp_0.sort()
    if len(lbp_1) % 2 == 0:
        print(lbp_1[int(len(lbp_1)/2) + 1])
    else:
        print(lbp_1[int(len(lbp_1)/2)])
    if len(lbp_0) % 2 == 0:
        print(lbp_0[int(len(lbp_0)/2) + 1])
    else:
        print(lbp_0[int(len(lbp_0)/2)])

    print('threshold:', threshold)
    print('acc:', rightNums / total_samples)
    print('FAR:', FA / spoof_samples_nums, 'FRR:', FR / live_samples_nums)
    print('Recall:', TP / live_samples_nums)


if __name__ == '__main__':
    print("Loading model...")
    classifier_predict = joblib.load('../model/424/424_train_best_10.1.pkl')
    model = pdx.deploy.Predictor(r'E:\Report\CelebA-Spoof\P0001-T0001_export_model\inference_model', use_gpu=False)
    detector = dlib.get_frontal_face_detector()
    print("Model loaded.")
    th = np.arange(0.5, 1, 0.1)
    for t in th:
        predict_CelebA_Spoof(r'F:\Datasets\images\test_300', threshold=t)
