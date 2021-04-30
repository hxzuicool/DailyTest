import os
import shutil
import cv2
import dlib

detector = dlib.get_frontal_face_detector()


def detect_face(path, fileName, isLive=False):
    global top, bottom, left, right

    img = cv2.imread(path)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 由于opencv中读取的图片为BGR通道，需要转为RGB通道，再利用detector
    faces = detector(RGB_img, 1)
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
        cv2.imwrite(r'E:\DataSets\3DMAD\face_images\live\\face_' + fileName, Roi_img)
    else:
        cv2.imwrite(r'E:\DataSets\3DMAD\face_images\spoof\\face_' + fileName, Roi_img)


if __name__ == '__main__':
    listdir_live = os.listdir(r'E:\DataSets\3DMAD\images\live')
    listdir_spoof = os.listdir(r'E:\DataSets\3DMAD\images\spoof')
    # for i in range(len(listdir)):
    #     basedir = os.path.basename(listdir[i])
    #     if os.path.exists(r'F:\Datasets\Images\train\\' + basedir + r'\spoof'):
    #         spoofListDir = os.listdir(r'F:\Datasets\Images\train' + basedir + r'\spoof')
    #         for j in range(len(spoofListDir)):
    #             fileName = os.path.basename(spoofListDir[j])
    #             if (os.path.splitext(fileName)[1] != '.jpg') and (os.path.splitext(fileName)[1] != '.png'):
    #                 continue
    #             print(r'F:\Datasets\Images\train\\' + basedir + r'\spoof\\' + fileName)
    #             detect_face(r'F:\Datasets\Images\train\\' + basedir + r'\spoof\\' + fileName, fileName, False)
    #
    #     if os.path.exists(r'F:\Datasets\Images\train\\' + basedir + r'\live'):
    #         liveListDif = os.listdir(r'F:\Datasets\Images\train\\' + basedir + r'\live')
    #         for j in range(len(liveListDif)):
    #             fileName = os.path.basename(liveListDif[j])
    #             if (os.path.splitext(fileName)[1] != '.jpg') and (os.path.splitext(fileName)[1] != '.png'):
    #                 continue
    #             print(r'F:\Datasets\Images\train\\' + basedir + r'\live\\' + fileName)
    #             detect_face(r'F:\Datasets\Images\train\\' + basedir + r'\live\\' + fileName, fileName, True)

    for i in range(len(listdir_live)):
        imageName = os.path.basename(listdir_live[i])
        detect_face(r'E:\DataSets\3DMAD\images\live\\' + imageName, imageName, True)

    for i in range(len(listdir_spoof)):
        imageName = os.path.basename(listdir_spoof[i])
        detect_face(r'E:\DataSets\3DMAD\images\spoof\\' + imageName, imageName, False)