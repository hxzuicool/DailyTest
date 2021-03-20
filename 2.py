import cv2
import numpy as np

img = cv2.imread('images/3.jpg')


def cv_imshow(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    cv_imshow('sobelx', sobelx)
