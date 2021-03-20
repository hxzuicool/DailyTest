import os
import shutil
import random


def copyFile(fileDir, tarDir, moveNum):
    filePath = os.listdir(fileDir)

    moveFiles = random.sample(filePath, moveNum)
    print(moveFiles)
    for name in moveFiles:
        shutil.copyfile(fileDir + name, tarDir + name)
        print(name)


if __name__ == '__main__':
    fileDir = 'E:\DataSets\CelebA_Spoof\Data\\new_train\live\\'
    tarDir = 'G:\\train\live\\'

    copyFile(fileDir, tarDir, 15000)
