import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0 / pixel_number
    # 发现bins必须写到257，否则255这个值只能分到[254,255)区间
    his, bins = np.histogram(gray, np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:  # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
        # print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    print(final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img


if __name__ == '__main__':
    # img_1 = Image.open('../images/Tom_Cruise_1.jpg')
    # img_1_gray = img_1.convert('L')
    # img_1_gray_otsu = otsu(img_1_gray)
    # plt.imshow(img_1_gray_otsu)
    # plt.show()

    img_1 = cv2.imread('../images/real_face.jpg')
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    # plt.imshow(img_1_gray)
    # plt.show()
    img_1_gray_otsu = otsu(img_1_gray)
    plt.subplot(121)
    plt.imshow(img_1_gray_otsu)
    plt.title("Real", fontsize='xx-large', fontweight='bold')

    img_0 = cv2.imread('../images/fake_face_3D.jpg')
    img_0_gray = cv2.cvtColor(img_0, cv2.COLOR_RGB2GRAY)
    img_0_gray_otsu = otsu(img_0_gray)
    plt.subplot(122)
    plt.imshow(img_0_gray_otsu)
    plt.title("Fake", fontsize='xx-large', fontweight='bold')
    plt.show()
