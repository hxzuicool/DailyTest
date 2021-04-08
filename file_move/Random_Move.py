import os
import shutil
import random


def copyFile(fileDir, tarDir, moveNum):
    filePath = os.listdir(fileDir)

    moveFiles = random.sample(filePath, moveNum)
    print(moveFiles)
    for name in moveFiles:
        shutil.copyfile(fileDir + '\\' + name, tarDir + '\\' + name)
        print(name)


if __name__ == '__main__':
    fileDir = r'E:\DataSets\CelebA_Spoof\New_Data\face_region\train_datasets\spoof'
    tarDir = r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize\test\spoof'

    copyFile(fileDir, tarDir, 500)
