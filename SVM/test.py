import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from skimage import feature as skif


if __name__ == '__main__':
    fake = cv2.imread('../images/Tom_Cruise_00.jpg', 0)
    image_bgr = cv2.imread('../images/Tom_Cruise.jpg')
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #
    # print(image_hsv.shape)
    # hist_local = np.zeros(256*3)
    # for i in range(3):
    #     img = image_hsv[:, :, i]
    #     lbp = skif.local_binary_pattern(img, 8, 1)
    #     max_bins = int(lbp.max() + 1)
    #     hist_local[i*256:(i+1)*256], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    #
    # cv2.calcHist()

    # f = np.fft.fft2(image_gray)
    # f0 = np.fft.fft2(fake)
    # fshift = np.fft.fftshift(f)
    # fshift0 = np.fft.fftshift(f0)
    # # 取绝对值：将复数变化成实数
    # # 取对数的目的为了将数据变化到较小的范围（比如0-255）
    # s1 = np.log(np.abs(f))
    # s2 = np.log(np.abs(fshift))
    # s3 = np.log(np.abs(f0))
    # s4 = np.log(np.abs(fshift0))
    #
    # plt.subplot(221), plt.imshow(s1, 'gray'), plt.title('original')
    # plt.subplot(222), plt.imshow(s2, 'gray'), plt.title('center')
    # plt.subplot(223), plt.imshow(s3, 'gray'), plt.title('original')
    # plt.subplot(224), plt.imshow(s4, 'gray'), plt.title('center')
    # plt.show()

    dft = cv2.dft(np.float32(image_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)
    # 设置高通滤波器

    rows, cols = image_gray.shape

    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

    mask = np.ones((rows, cols, 2), np.uint8)

    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # 掩膜图像和频谱图像乘积

    f = fshift * mask

    # 傅里叶逆变换

    ishift = np.fft.ifftshift(f)

    iimg = cv2.idft(ishift)

    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    # 显示原始图像和高通滤波处理图像

    plt.subplot(121), plt.imshow(image_gray, 'gray'), plt.title('Original Image')

    plt.axis('off')

    plt.subplot(122), plt.imshow(res, 'gray'), plt.title('High Pass Filter Image')

    plt.axis('off')

    plt.show()
