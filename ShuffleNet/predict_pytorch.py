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


def combine_global_lbp(path):
    global top, bottom, left, right
    total_samples = 0
    Acc = 0
    RightNums = 0
    hist = np.zeros((256 * 3))
    for live_or_spoof in ['live', 'spoof']:
        folderPath = path + '\\' + live_or_spoof
        imagesName_list = os.listdir(folderPath)
        total_samples = total_samples + len(imagesName_list)
        for idx in range(len(imagesName_list)):
            filePath = folderPath + '\\' + imagesName_list[idx]
            img = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2RGB).astype('float32')
            result = model.predict(img)
            if result[0]['category'] == live_or_spoof:
                RightNums = RightNums + 1
                print(imagesName_list[idx], '  ', result[0]['category'], '  ', result[0]['score'])

            img_ = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2RGB)
            faces = detector(img_, 1)
            if len(faces) == 0:
                continue
            left = faces[0].left()
            right = faces[0].right()
            top = faces[0].top()
            bottom = faces[0].bottom()
            if left < 0 or right < 0 or top < 0 or bottom < 0:
                continue
            img_face = img_[top:bottom, left:right]
            img_face = cv2.resize(img_face, (224, 224))
            # plt.imshow(img_face)
            # plt.show()
            img_face_hsv = cv2.cvtColor(img_face, cv2.COLOR_RGB2HSV)

            j = 0
            for face_img_channel in cv2.split(img_face_hsv):
                lbp = local_binary_pattern(face_img_channel, 8, 1)
                max_bins = int(lbp.max() + 1)
                histogram, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
                print(histogram.shape)
                hist[256 * j:256 * (j + 1)] = histogram
                j = j + 1
            predict_lbp = classifier_predict.predict(hist.reshape(1, -1))
            predict_value = classifier_predict.decision_function(hist.reshape(1, -1))

            if live_or_spoof == 'live':
                if result[0]['category'] == live_or_spoof:
                    RightNums = RightNums + 1


def combine_global_lbp_WMCA(path):
    global top, bottom, left, right
    total_samples = 0
    index = 0
    RightNums_CNN = 0
    rightNums_lbp = 0
    rightNums = 0
    FA = 0
    FR = 0
    TP = 0
    live_samples_nums = 0
    spoof_samples_nums = 0
    hist = np.zeros((256 * 3))
    for live_or_spoof in ['live', 'spoof']:
        folderPath = path + '\\' + live_or_spoof
        imagesName_list = os.listdir(folderPath)
        if live_or_spoof == 'live':
            live_samples_nums = len(imagesName_list)
        else:
            spoof_samples_nums = len(imagesName_list)
        total_samples = total_samples + len(imagesName_list)
        for idx in range(len(imagesName_list)):
            filePath = folderPath + '\\' + imagesName_list[idx]
            img_ = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2RGB)
            img = img_.astype('float32')
            predict_shuffleNet = model.predict(img)

            faces = detector(img_, 1)
            if len(faces) == 0:
                if live_or_spoof == 'spoof':
                    rightNums = rightNums + 1
                    rightNums_lbp = rightNums_lbp + 1
                elif predict_shuffleNet[0]['category'] == 'live':
                    TP = TP + 1
                    rightNums = rightNums + 1
                    RightNums_CNN = RightNums_CNN + 1
                index = index + 1
                continue
            left = faces[0].left()
            right = faces[0].right()
            top = faces[0].top()
            bottom = faces[0].bottom()
            if left < 0 or right < 0 or top < 0 or bottom < 0:
                if live_or_spoof == 'spoof':
                    rightNums = rightNums + 1
                    rightNums_lbp = rightNums_lbp + 1
                elif predict_shuffleNet[0]['category'] == 'live':
                    TP = TP + 1
                    rightNums = rightNums + 1
                    RightNums_CNN = RightNums_CNN + 1
                index = index + 1
                continue
            img_face = img_[top:bottom, left:right]

            img_face_torch = trans(img_face) * 255
            img_face_hsv = cv2.cvtColor(img_face_torch.numpy().transpose((1, 2, 0)), cv2.COLOR_RGB2HSV)

            for j, face_img_channel in enumerate(cv2.split(img_face_hsv)):
                lbp = local_binary_pattern(face_img_channel, 8, 1)
                max_bins = int(lbp.max() + 1)
                histogram, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
                hist[256 * j:256 * (j + 1)] = histogram

            predict_lbp = classifier_predict.predict(hist.reshape(1, -1))
            # predict_value = classifier_predict.decision_function(hist.reshape(1, -1))
            # print(predict_lbp, predict_value)

            if predict_shuffleNet[0]['category'] == live_or_spoof:
                RightNums_CNN = RightNums_CNN + 1
                # print(imagesName_list[idx], '  ', predict_shuffleNet[0]['category'], '  ',
                #       predict_shuffleNet[0]['score'])

            if (live_or_spoof == 'live' and predict_lbp == 0) or (live_or_spoof == 'spoof' and predict_lbp == 1):
                rightNums_lbp = rightNums_lbp + 1

            # if live_or_spoof == 'live' and (predict_shuffleNet[0]['category'] == 'live' and predict_lbp == 0):
            if live_or_spoof == 'live' and ((predict_lbp == 0 and predict_shuffleNet[0]['category'] == 'live') or
                                            (predict_lbp == 1 and predict_shuffleNet[0]['category'] == 'live')):
                TP = TP + 1
                rightNums = rightNums + 1

            if live_or_spoof == 'spoof' and ((predict_shuffleNet[0]['category'] == 'live' and predict_lbp == 1) or
                                             (predict_shuffleNet[0]['category'] == 'spoof' and predict_lbp == 0) or
                                             (predict_shuffleNet[0]['category'] == 'spoof' and predict_lbp == 1)):
                rightNums = rightNums + 1
            else:
                FA = FA + 1

            if live_or_spoof == 'live' and ((predict_shuffleNet[0]['category'] == 'live' and predict_lbp == 1) or
                                            (predict_shuffleNet[0]['category'] == 'spoof' and predict_lbp == 0) or
                                            (predict_shuffleNet[0]['category'] == 'spoof' and predict_lbp == 1)):
                FR = FR + 1
            print(index, '/', total_samples, live_or_spoof, 'lbp:', predict_lbp, 'shuffleNet:',
                  predict_shuffleNet[0]['category'], 'acc:', rightNums / (index + 1))
            index = index + 1
    print('CNN_Acc:', RightNums_CNN / total_samples, 'lbp_Acc:', rightNums_lbp / total_samples)
    print('total acc:', rightNums / total_samples)
    print('FAR:', FA / spoof_samples_nums, 'FRR:', FR / live_samples_nums)
    print('Recall:', TP / live_samples_nums)


def t1():
    hist = np.zeros((256 * 3))
    img = cv2.imread(r'F:\Datasets\temp_test\spoof\face_509453.png')
    # torch_img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    torch_img = trans(img)
    torch_img = torch_img * 255
    img_hsv = cv2.cvtColor(torch_img.numpy().transpose((1, 2, 0)), cv2.COLOR_BGR2HSV)
    for i, img_channel in enumerate(cv2.split(img_hsv)):
        lbp = skif.local_binary_pattern(img_channel, 8, 1, 'default')
        max_bins = int(lbp.max() + 1)
        histogram, _ = np.histogram(lbp, density=True, bins=int(max_bins), range=(0, max_bins))
        hist[i * 256:(i + 1) * 256] = histogram

    predict = classifier_predict.predict(hist.reshape(1, -1))
    print(predict)


def combine_global_lbp_CelebA_Spoof(path):
    global top, bottom, left, right
    index = 0
    RightNums_CNN = 0
    rightNums_lbp = 0
    rightNums = 0
    FA = 0
    FR = 0
    TP = 0
    live_samples_nums = 0
    spoof_samples_nums = 0
    hist = np.zeros((256 * 3))
    for live_or_spoof in ['live', 'spoof']:
        folderPath = path + '\\' + live_or_spoof
        imagesName_list = os.listdir(folderPath)
        if live_or_spoof == 'live':
            live_samples_nums = len(imagesName_list)
        else:
            spoof_samples_nums = len(imagesName_list)
        total_samples = live_samples_nums + spoof_samples_nums

        for idx in range(len(imagesName_list)):
            filePath = folderPath + '\\' + imagesName_list[idx]
            predict_shuffleNet = model.predict(filePath, topk=2)
            # print('{:.7f}'.format(predict_shuffleNet[1]['score']))

            img_ = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2RGB)
            faces = detector(img_, 1)

            if len(faces) == 0:
                if live_or_spoof == 'spoof' and predict_shuffleNet[0]['category'] == 'spoof':
                    rightNums = rightNums + 1
                    rightNums_lbp = rightNums_lbp + 1
                    RightNums_CNN = RightNums_CNN + 1
                elif live_or_spoof == 'live' and predict_shuffleNet[0]['category'] == 'live':
                    TP = TP + 1
                    rightNums = rightNums + 1
                    RightNums_CNN = RightNums_CNN + 1
                index = index + 1
                continue
            left = faces[0].left()
            right = faces[0].right()
            top = faces[0].top()
            bottom = faces[0].bottom()
            if left < 0 or right < 0 or top < 0 or bottom < 0:
                if live_or_spoof == 'spoof' and predict_shuffleNet[0]['category'] == 'spoof':
                    rightNums = rightNums + 1
                    rightNums_lbp = rightNums_lbp + 1
                    RightNums_CNN = RightNums_CNN + 1
                elif live_or_spoof == 'live' and predict_shuffleNet[0]['category'] == 'live':
                    TP = TP + 1
                    rightNums = rightNums + 1
                    RightNums_CNN = RightNums_CNN + 1
                index = index + 1
                continue
            img_face = img_[top:bottom, left:right]

            img_face_torch = trans(img_face) * 255
            img_face_hsv = cv2.cvtColor(img_face_torch.numpy().transpose((1, 2, 0)), cv2.COLOR_RGB2HSV)

            for j, face_img_channel in enumerate(cv2.split(img_face_hsv)):
                lbp = local_binary_pattern(face_img_channel, 8, 1)
                max_bins = int(lbp.max() + 1)
                histogram, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
                hist[256 * j:256 * (j + 1)] = histogram

            predict_lbp = classifier_predict.predict(hist.reshape(1, -1))
            # predict_value = classifier_predict.decision_function(hist.reshape(1, -1))
            # print(predict_lbp, predict_value)

            if predict_shuffleNet[0]['category'] == live_or_spoof:
                RightNums_CNN = RightNums_CNN + 1

            if (live_or_spoof == 'live' and predict_lbp == 0) or (live_or_spoof == 'spoof' and predict_lbp == 1):
                rightNums_lbp = rightNums_lbp + 1

            if live_or_spoof == 'live' and predict_shuffleNet[0]['category'] == 'live':
                TP = TP + 1
                rightNums = rightNums + 1
            elif live_or_spoof == 'live':
                FR = FR + 1

            if live_or_spoof == 'spoof' and ((predict_shuffleNet[0]['category'] == 'live' and predict_lbp == 1) or
                                             (predict_shuffleNet[0]['category'] == 'spoof' and predict_lbp == 0) or
                                             (predict_shuffleNet[0]['category'] == 'spoof' and predict_lbp == 1)):
                rightNums = rightNums + 1
            elif live_or_spoof == 'spoof':
                FA = FA + 1

            print(index, '/', total_samples, live_or_spoof, 'lbp:', predict_lbp, 'shuffleNet:',
                  predict_shuffleNet[0]['category'], predict_shuffleNet[0]['score'], 'acc:', rightNums / (index + 1))
            index = index + 1
    print('CNN_Acc:', RightNums_CNN / total_samples, 'lbp_Acc:', rightNums_lbp / total_samples)
    print('total acc:', rightNums / total_samples)
    print('FAR:', FA / spoof_samples_nums, 'FRR:', FR / live_samples_nums)
    print('Recall:', TP / live_samples_nums)


if __name__ == '__main__':
    print("Loading model...")
    classifier_predict = joblib.load('../model/424/424_train_best_10.1.pkl')
    model = pdx.deploy.Predictor(r'E:\Report\CelebA-Spoof\P0001-T0001_export_model\inference_model', use_gpu=False)
    detector = dlib.get_frontal_face_detector()
    print("Model loaded.")
    combine_global_lbp_CelebA_Spoof(r'F:\Datasets\images\test_sub')
