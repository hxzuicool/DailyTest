import os
import shutil
import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np

train_list = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\test')

print(len(train_list))


def move01():
    for i in range(0, len(train_list)):
        dir = os.path.basename(train_list[i])

        if os.path.exists(r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + r'\spoof'):
            list_spoofing = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + r'\spoof')
            for k in range(0, len(list_spoofing)):
                imgName = os.path.basename(list_spoofing[k])
                if (os.path.splitext(imgName)[1] != '.jpg') and (os.path.splitext(imgName)[1] != '.png'):
                    continue

                oldName = r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + r'\spoof\\' + imgName
                newName = r'F:\Datasets\Images\test\spoof\\' + imgName

                shutil.copyfile(oldName, newName)
                print(newName)

        if os.path.exists(r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + r'\live'):
            list_living = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + r'\live')
            for j in range(0, len(list_living)):
                imgName = os.path.basename(list_living[j])
                if os.path.splitext(imgName)[1] != '.jpg' and os.path.splitext(imgName)[1] != '.png':
                    continue

                oldName = r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + r'\live\\' + imgName
                newName = r'F:\Datasets\Images\test\live\\' + imgName

                shutil.copyfile(oldName, newName)
                print(newName)


def move02():
    filePath = r'E:\DataSets\WMCA_RGB\preprocessed-face-station_RGB'
    index = 1

    with open(r'E:\DataSets\WMCA_RGB\documentation\bonafide_illustration_files.csv') as files:
        for line in files:
            filename = line.strip('\n')
            file = h5py.File(filePath + '\\' + filename, 'r')
            for key in file.keys():
                f = file[key]
                # print(f['array'])
                cv2.imwrite(r'F:\Datasets\WMCA_RGB\live\\' + str(index) + '.jpg',
                            cv2.cvtColor(f['array'][:].transpose((1, 2, 0)), cv2.COLOR_BGR2RGB))
                index = index + 1


def readCSV():
    with open(r'E:\DataSets\WMCA_CDIT\documentation\attack_illustration_files.csv') as file:
        for line in file:
            print(line.split('/'))


if __name__ == '__main__':
    move02()
