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
image_1 = cv2.imread(r'..\images\Tom_Cruise.jpg', 0)
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


def image_hist(image):
    color = ('Y', 'Cb', 'Cr')
    print(image)
    for i, color in enumerate(color):  # 绘制每一个颜色对应的直方图，从容器中进行迭代
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.subplot(1, 3, i+1)
        plt.plot(hist)
        plt.xlim([0, 256])
    plt.show()


if __name__ == '__main__':
    img = showYCbCr(image_0)
    # img_lbp = LBP(img)
    # histogram(img_lbp.astype(np.uint8))
    # plt.imshow(img_lbp, cmap='gray')
    # plt.show()
    # cv2.imwrite('../images/001450_Cr_LBP.jpg', img_lbp)
    # image_hist(img)

    # Y, Cb, Cr = cv2.split(img)
    # YCbCr = (Y, Cb, Cr)
    # plt.figure(figsize=(15, 4))
    # for i, img_YCbCr in enumerate(YCbCr):
    #     lbp_img = local_binary_pattern(img_YCbCr, n_points, radius)
    #     calc_hist = cv2.calcHist([lbp_img.astype(np.uint8)], [0], None, [256], [0, 255])
    #     plt.subplot(1, 3, i+1)
    #     plt.plot(calc_hist)
    #     plt.xlim([0, 255])
    #     plt.axis('off')
    #
    # plt.show()

    histogram(LBP(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)).astype(np.uint8))