import numpy as np
import cv2, os, math, os.path, glob, random

g_mapping = [
    0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10,
    11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
    16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
    22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
    29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
    36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
    42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
    47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57]


def loadImageSet(folder, sampleCount=5):
    trainData = []
    testData = []
    yTrain = []
    yTest = []
    for k in range(1, 41):
        folder2 = os.path.join(folder, 's%d' % k)
        data = [cv2.imread(d.encode('gbk'), 0) for d in glob.glob(os.path.join(folder2, '*.pgm'))]
        sample = random.sample(range(10), sampleCount)
        trainData.extend([data[i] for i in range(10) if i in sample])
        testData.extend([data[i] for i in range(10) if i not in sample])
        yTest.extend([k] * (10 - sampleCount))
        yTrain.extend([k] * sampleCount)
    return trainData, testData, np.array(yTrain), np.array(yTest)


def LBP(I, radius=2, count=8):  # 得到图像的LBP特征
    dh = np.round([radius * math.sin(i * 2 * math.pi / count) for i in range(count)])
    dw = np.round([radius * math.cos(i * 2 * math.pi / count) for i in range(count)])

    height, width = I.shape
    lbp = np.zeros(I.shape, dtype=np.int)
    I1 = np.pad(I, radius, 'edge')
    for k in range(count):
        h, w = radius + dh[k], radius + dw[k]
        lbp += ((I > I1[h:h + height, w:w + width]) << k)
    return lbp


def calLbpHistogram(lbp, hCount=7, wCount=5, maxLbpValue=255):  # 分块计算lbp直方图
    height, width = lbp.shape
    res = np.zeros((hCount * wCount, max(g_mapping) + 1), dtype=np.float)
    assert (maxLbpValue + 1 == len(g_mapping))

    for h in range(hCount):
        for w in range(wCount):
            blk = lbp[height * h / hCount:height * (h + 1) / hCount, width * w / wCount:width * (w + 1) / wCount]
            hist1 = np.bincount(blk.ravel(), minlength=maxLbpValue)

            hist = res[h * wCount + w, :]
            for v, k in zip(hist1, g_mapping):
                hist[k] += v
            hist /= hist.sum()
    return res


def main(folder=u'E:/迅雷下载/faceProcess/att_faces'):
    trainImg, testImg, yTrain, yTest = loadImageSet(folder)

    xTrain = np.array([calLbpHistogram(LBP(d)).ravel() for d in trainImg])
    xTest = np.array([calLbpHistogram(LBP(d)).ravel() for d in testImg])

    lsvc = cv2.SVM()  # 支持向量机方法
    svm_params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=2.67, gamma=5.383)
    lsvc.train(np.float32(xTrain), np.float32(yTrain), params=svm_params)
    lsvc_y_predict = np.array([lsvc.predict(d) for d in np.float32(xTest)])
    print(u'支持向量机识别率', (lsvc_y_predict == np.array(yTest)).mean())


if __name__ == '__main__':
    main()
