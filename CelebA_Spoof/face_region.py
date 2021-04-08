import os
import shutil
import cv2


def cropFace():
    listdir = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\train')
    print(len(listdir))
    for i in range(len(listdir)):
        basedir = os.path.basename(listdir[i])
        if os.path.exists(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\spoof'):
            spoofListDir = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\spoof')
            for j in range(len(spoofListDir)):
                fileName = os.path.basename(spoofListDir[j])
                if (os.path.splitext(fileName)[1] != '.jpg') and (os.path.splitext(fileName)[1] != '.png'):
                    continue
                txt = open(
                    r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\spoof\\' + os.path.splitext(fileName)[
                        0] + '_BB.txt')
                text = txt.readline()
                txtArray = text.split(' ')
                if float(txtArray[4]) < 0.5:
                    print(txtArray)
                img = cv2.imread(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\spoof\\' + fileName)
                img_height = img.shape[0]
                img_width = img.shape[1]
                x1 = int(int(txtArray[0]) * (img_width / 224))
                y1 = int(int(txtArray[1]) * (img_height / 224))
                x2 = int(int(txtArray[2]) * (img_width / 224))
                y2 = int(int(txtArray[3]) * (img_height / 224))
                cropimg = img[x1: x1+x2, y1: y1+y2]
                print(r'E:\DataSets\CelebA_Spoof\New_Data\face_region\spoof\\' + os.path.splitext(fileName)[
                    0] + '_crop.jpg')
                cv2.imwrite(r'E:\DataSets\CelebA_Spoof\New_Data\face_region\spoof\\' + os.path.splitext(fileName)[
                    0] + '_crop.jpg', cropimg)

        if os.path.exists(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\live'):
            spoofListDir = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\live')
            for j in range(len(spoofListDir)):
                fileName = os.path.basename(spoofListDir[j])
                if (os.path.splitext(fileName)[1] != '.jpg') and (os.path.splitext(fileName)[1] != '.png'):
                    continue
                txt = open(
                    r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\live\\' + os.path.splitext(fileName)[
                        0] + '_BB.txt')
                text = txt.readline()
                txtArray = text.split(' ')
                if float(txtArray[4]) < 0.8:
                    continue
                img = cv2.imread(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\live\\' + fileName)
                img_height = img.shape[0]
                img_width = img.shape[1]
                x1 = int(int(txtArray[0]) * (img_width / 224))
                y1 = int(int(txtArray[1]) * (img_height / 224))
                x2 = int(int(txtArray[2]) * (img_width / 224))
                y2 = int(int(txtArray[3]) * (img_height / 224))
                cropimg = img[x1: x1+x2, y1: y1+y2]
                cv2.imwrite(r'E:\DataSets\CelebA_Spoof\New_Data\face_region\live\\' + os.path.splitext(fileName)[
                    0] + '_crop.jpg', cropimg)


if __name__ == '__main__':
    cropFace()
