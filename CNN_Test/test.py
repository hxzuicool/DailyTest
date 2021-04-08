import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show("直方图")


def image_hist(image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):  # 绘制每一个颜色对应的直方图，从容器中进行迭代
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


"""images: 输入的图像。
   channels: 需要统计直方图的第几通道
   mask: 可选的操作掩膜。
   histSize: 指的是直方图分成多少个区间，就是 bin的个数
   ranges: 每个维度中bin的取值范围，统计像素值的区间
"""
src = cv.imread("../images/Tom_Cruise.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
# plot_demo(src)
image_hist(src)

cv.waitKey(0)
cv.destroyAllWindows()
