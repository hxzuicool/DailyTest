# 将数据集分为戴眼镜和不戴眼镜

import os
import shutil

nof = open("./data/CelebA/noGlass.txt")
hasf = open("./data/CelebA/glass.txt")

noLine = nof.readline()
hasLine = hasf.readline()

list = os.listdir("E:\DataSets\Images_CelebA\img_align_celeba")
hasGo = True
noGo = True
if not os.path.exists('./faces/train/noGlass'):
    os.mkdir('./faces/train/noGlass')
    print('noGlass文件夹创建成功！')

if not os.path.exists('./faces/train/hasGlass'):
    os.mkdir('./faces/train/hasGlass')
    print('hasGlass文件夹创建成功！')

for i in range(0, len(list)):
    imgName = os.path.basename(list[i])
    if os.path.splitext(imgName)[1] != ".jpg":
        continue

    noArray = noLine.split()
    if len(noArray) < 1:
        noGo = False
    hasArray = hasLine.split()
    if len(hasArray) < 1:
        hasGo = False

    if noGo and (imgName == noArray[0]):
        oldname = "E:\DataSets\Images_CelebA\img_align_celeba\\" + imgName
        newname = "./faces/train/noGlass/" + imgName
        shutil.copyfile(oldname, newname)
        noLine = nof.readline()
    elif hasGo and (imgName == hasArray[0]):
        oldname = "E:\DataSets\Images_CelebA\img_align_celeba\\" + imgName
        newname = "./faces/train/hasGlass/" + imgName
        shutil.copyfile(oldname, newname)
        hasLine = hasf.readline()

    print('i:', i)

nof.close()
hasf.close()
