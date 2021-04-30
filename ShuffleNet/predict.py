import paddlex as pdx
import cv2
import os
import shutil
import dlib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np
import joblib


def loadmodel():
    # 模型加载, 请将path_to_model替换为你的模型导出路径
    # 可使用 mode = pdx.load_model('path_to_model') 加载
    # 而使用Predictor方式加载模型，会对模型计算图进行优化，预测速度会更快
    print("Loading model...")
    model = pdx.deploy.Predictor(r'E:\Report\P0001-T0001_export_model\inference_model', use_gpu=False)
    print("Model loaded.")
    return model


# 模型预测, 可以将图片替换为你需要替换的图片地址
# 使用Predictor时，刚开始速度会比较慢，参考此issue
# https://github.com/PaddlePaddle/PaddleX/issues/116
# im = cv2.imread('../images/001450.jpg')
# im = im.astype('float32')

# result = model.predict(im)

# for key in result[0].keys():
#     print(key)
# 输出分类结果
# if model.model_type == "classifier":
#     print(result)

# 可视化结果, 对于检测、实例分割务进行可视化
# if model.model_type == "detector":
# threshold用于过滤低置信度目标框
# 可视化结果保存在当前目录
# pdx.det.visualize(im, result, threshold=0.5, save_dir='./')

# 可视化结果, 对于语义分割务进行可视化
# if model.model_type == "segmenter":
# weight用于调整结果叠加时原图的权重
# 可视化结果保存在当前目录
# pdx.seg.visualize(im, result, weight=0.0, save_dir='./')

def predict_list(folderPath, live_or_spoof):
    folderPath = folderPath + '\\' + live_or_spoof
    imagesName_list = os.listdir(folderPath)
    print(len(imagesName_list))
    falseNum = 0
    for idx in range(len(imagesName_list)):
        filePath = folderPath + '\\' + imagesName_list[idx]
        img = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2RGB).astype('float32')
        result = model.predict(img)
        # print(result[0]['category'], '  ', result[0]['score'])
        if result[0]['category'] != live_or_spoof:
            falseNum = falseNum + 1
            print(imagesName_list[idx], '  ', result[0]['category'], '  ', result[0]['score'])
            # shutil.copyfile(filePath, r'F:\Datasets\images\test_error\live\\' + imagesName_list[idx])
    print('错误率：', falseNum / len(imagesName_list))


def predict_one():
    imagePath = r'F:\Datasets\images\test\spoof\494457.png'
    image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB).astype('float32')
    result = model.predict(image)
    print(result[0]['category'], '  ', result[0]['score'])


def predict_face_lbp(folderPath, live_or_spoof):
    globalImagePath = folderPath + '\\' + live_or_spoof
    imagesName = os.listdir(globalImagePath)
    global top, bottom, left, right
    falseNums = 0
    classifier_predict = joblib.load('../model/424/424_train_best_10.1.pkl')
    hist = np.zeros((256 * 3))
    for i in range(len(imagesName)):
        img = cv2.imread(globalImagePath + '\\' + imagesName[i])
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(RGB_img, 1)
        if len(faces) == 0:
            falseNums = falseNums + 1
            continue
        # for idx, face in enumerate(faces):
        #     left = face.left()
        #     right = face.right()
        #     top = face.top()
        #     bottom = face.bottom()
        left = faces[0].left()
        right = faces[0].right()
        top = faces[0].top()
        bottom = faces[0].bottom()

        if left < 0 or right < 0 or top < 0 or bottom < 0:
            falseNums = falseNums + 1
            continue
        face_img = RGB_img[top:bottom, left:right]

        face_img = cv2.resize(face_img, (224, 224))
        face_img_hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
        j = 0
        for face_img_channel in cv2.split(face_img_hsv):
            lbp = local_binary_pattern(face_img_channel, 8, 1)
            max_bins = int(lbp.max() + 1)
            histogram, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
            # plt.plot(histogram)
            # plt.xlim([0, 256])
            # plt.show()
            hist[256 * j:256 * (j + 1)] = histogram
            j = j + 1
        predict = classifier_predict.predict(hist.reshape(1, -1))
        predict_value = classifier_predict.decision_function(hist.reshape(1, -1))
        print(predict, predict_value)
        if (live_or_spoof == 'live' and predict == 0) or (live_or_spoof == 'spoof' and predict == 1):
            falseNums = falseNums + 1
    print('acc:', falseNums / len(imagesName))


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
            predict = classifier_predict.predict(hist.reshape(1, -1))
            predict_value = classifier_predict.decision_function(hist.reshape(1, -1))

            if live_or_spoof == 'live':
                if result[0]['category'] == live_or_spoof:
                    RightNums = RightNums + 1


if __name__ == '__main__':
    # category_id, category, score
    # model = loadmodel()
    # predict_list(r'F:\Datasets\temp_test', 'live')
    classifier_predict = joblib.load('../model/424/424_train_best_10.1.pkl')
    # detector = dlib.get_frontal_face_detector()
    predict_face_lbp(r'F:\Datasets\temp_test', live_or_spoof='spoof')
    # combine_global_lbp(r'F:\Datasets\images\test')
