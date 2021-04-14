import os
import cv2


def resize_image(filepath, tarpath, filename, height_dst, width_dst):
    img = cv2.imread(filepath + '\\' + filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    fx = width_dst / width
    fy = height_dst / height
    resize_img = cv2.resize(img, (0, 0), fx=fx, fy=fy)

    cv2.imwrite(tarpath + '\\' + filename, resize_img)
    return resize_img


if __name__ == '__main__':
    filepath = r'F:\Datasets\face_region_random_resize\train\live'
    tarpath = filepath  # resize的图片保存在原文件夹
    listdir = os.listdir(filepath)
    for i in range(len(listdir)):
        filename = os.path.basename(listdir[i])

        resize_image(filepath, tarpath, filename, 256, 256)
