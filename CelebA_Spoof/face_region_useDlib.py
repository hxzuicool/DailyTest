import os
import shutil
import cv2
import dlib


def detect_face(path, fileName, isLive=False):
    global top, bottom, left, right
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(path)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 由于opencv中读取的图片为BGR通道，需要转为RGB通道，再利用detector
    faces = detector(RGB_img, 1)
    print(len(faces))
    if len(faces) == 0:
        return
    for idx, face in enumerate(faces):
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        print(left, right, top, bottom)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    Roi_img = img[top:bottom, left:right]
    if isLive:
        cv2.imwrite(r'E:\DataSets\CelebA_Spoof\New_Data\face_region\live\\' + fileName + '_crop.jpg', Roi_img)
    else:
        cv2.imwrite(r'E:\DataSets\CelebA_Spoof\New_Data\face_region\spoof\\' + fileName + '_crop.jpg', Roi_img)


if __name__ == '__main__':
    listdir = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\train')
    for i in range(len(listdir)):
        basedir = os.path.basename(listdir[i])
        if os.path.exists(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\spoof'):
            spoofListDir = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\spoof')
            for j in range(len(spoofListDir)):
                fileName = os.path.basename(spoofListDir[j])
                if (os.path.splitext(fileName)[1] != '.jpg') and (os.path.splitext(fileName)[1] != '.png'):
                    continue
                print(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\spoof\\' + fileName)
                detect_face(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\spoof\\' + fileName, fileName, False)

        if os.path.exists(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\live'):
            liveListDif = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\live')
            for j in range(len(liveListDif)):
                fileName = os.path.basename(liveListDif[j])
                if (os.path.splitext(fileName)[1] != '.jpg') and (os.path.splitext(fileName)[1] != '.png'):
                    continue
                print(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\live\\' + fileName)
                detect_face(r'E:\DataSets\CelebA_Spoof\Data\train\\' + basedir + r'\live\\' + fileName, fileName, True)
