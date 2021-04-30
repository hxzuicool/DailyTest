import os
import shutil
import random
import cv2


def copyFile(fileDir, tarDir, moveNum):
    live_or_spoof = ['live', 'spoof']
    for RF in live_or_spoof:
        imagesPath = fileDir + '\\' + RF
        tarPath = tarDir + '\\' + RF
        imagesName = os.listdir(imagesPath)

        nums = int(len(imagesName) * 0.2)
        moveFiles = random.sample(imagesName, 10000)
        print(moveFiles)
        for name in moveFiles:
            shutil.copyfile(imagesPath + '\\' + name, tarPath + '\\' + name)
            print(name)


def filter_image_size(fileDir):
    nums = 0
    for i in ['live', 'spoof']:
        path = fileDir + '\\' + i
        images_name = os.listdir(path)
        for j in range(len(images_name)):
            img = cv2.imread(path + '\\' + images_name[j])
            if img.shape[0] < 100 or img.shape[1] < 100:
                print(img.shape)
                nums = nums + 1
                os.remove(path + '\\' + images_name[j])
    print(nums)


if __name__ == '__main__':
    fileDir = r'F:\Datasets\images\test'
    tarDir = r'F:\Datasets\images\test_10000'
    nums = int(len(os.listdir(fileDir)) / 2)
    copyFile(fileDir, tarDir, 10)

    # filter_image_size(r'G:\hxzuicool\mixture')
