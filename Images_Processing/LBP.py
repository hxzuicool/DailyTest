from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io, data_dir, filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# settings for LBP
radius = 1  # LBP算法中范围半径的取值
n_points = 8 * radius  # 领域像素点数

# 读取图像
image_1 = cv2.imread(r'..\images\Tom_Cruise.jpg')
image_0 = cv2.imread(r'../images/001450.jpg_crop.jpg')


# 显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
# image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.subplot(111)
# plt.imshow(image1)

# image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# plt.subplot(111)
# plt.imshow(image, plt.cm.gray)

# image_1 = np.array(image_1)
# image_0 = np.array(image_0)
# lbp_1 = local_binary_pattern(image_1, n_points, radius)
# lbp_0 = local_binary_pattern(image_0, n_points, radius)
# plt.subplot(121)
# plt.imshow(lbp_1, plt.cm.gray)
#
# plt.subplot(122)
# plt.imshow(lbp_0, plt.cm.gray)


# edges = filters.sobel(image)
# plt.subplot(111)
# plt.imshow(edges, plt.cm.gray)

# hog = feature.hog(image_1)
# print(hog)
# plt.subplot(111)
# plt.imshow(hog)

# plt.hist(hog)

def showYCbCr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # img_YCbCr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # Y, Cb, Cr = cv2.split(img)
    return img


def LBP(img):
    lbp = local_binary_pattern(img, n_points, radius)
    return lbp


def histogram(img):
    calc_hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    print(calc_hist.shape)
    plt.plot(calc_hist)
    plt.xlim([0, 255])
    # plt.imshow(calc_hist)
    # plt.hist(img.ravel(), 255, [0, 255], color='black')
    # plt.axis('off')
    plt.show()


def image_hist_cv2(image):
    color = ('Y', 'Cb', 'Cr')
    # print(image)
    for i, color in enumerate(color):  # 绘制每一个颜色对应的直方图，从容器中进行迭代
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.subplot(1, 3, i + 1)
        plt.plot(hist)
        plt.xlim([0, 256])
    plt.show()


def image_hist_np(image):
    image_channel = cv2.split(image)
    for i, color in enumerate(image_channel):
        img_lbp = LBP(image_channel[i])
        max_bins = int(img_lbp.max() + 1)
        hist_local, _ = np.histogram(img_lbp, density=True, bins=max_bins, range=(0, max_bins))
        plt.subplot(1, 3, i + 1)
        plt.plot(hist_local, color='black')
        plt.xlim([0, 256])
        plt.axis('off')
    plt.show()


def histogram_all(img):
    for i in range(3):
        img_lbp = LBP(cv2.split(img)[i]).astype(np.uint8)
        calc_hist = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
        # max_bins = int(img_lbp.max() + 1)
        # # hist size:256
        # hist_local, _ = np.histogram(img_lbp, density=True, bins=max_bins, range=(0, max_bins))
        for j in range(len(calc_hist)):
            hist[i * 256 + j][0] = calc_hist[j]
    plt.plot(hist, color='black')
    plt.xlim([0, 256 * 3])
    # plt.imshow(calc_hist)
    # plt.hist(img.ravel(), 255, [0, 255], color='black')
    # plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # img = showYCbCr(image_0)
    img = cv2.cvtColor(image_1, cv2.COLOR_BGR2HSV)
    hist = np.zeros((256 * 3, 1))
    # image_hist(img)

    bgr = cv2.imread('../images/env/000897_face.jpg')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    hist_ = np.zeros(256 * 3)
    for i, color in enumerate(cv2.split(hsv)):
        img_lbp = LBP(cv2.split(hsv)[i])
        max_bins = int(img_lbp.max() + 1)
        hist_local, _ = np.histogram(img_lbp, density=True, bins=max_bins, range=(0, max_bins))
        hist_[256 * i:(i + 1) * 256] = hist_local
    plt.plot(hist_, color='black')
    plt.xlim([0, 256 * 3])
    plt.axis('off')
    plt.show()
