import os
import shutil
import cv2


def movefile(alive='live'):
    fileList = os.listdir(r'E:\DataSets\CelebA_Spoof\New_Data\face_region\\' + alive)
    for i in range(0, len(fileList)):
        srcPath = r'E:\DataSets\CelebA_Spoof\New_Data\face_region\\' + alive + '\\' + fileList[i]

        dstPath = r'E:\DataSets\CelebA_Spoof\New_Data\face_region_100\\' + alive
        img = cv2.imread(srcPath)

        if img.shape[0] < 110 or img.shape[1] < 110:
            shutil.move(srcPath, dstPath)


if __name__ == '__main__':
    movefile('live')
    movefile('spoof')