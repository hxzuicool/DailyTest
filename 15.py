import os
import shutil

list_hasGlass = os.listdir('faces/train/hasGlass')
print(len(list_hasGlass))

list_noGlass = os.listdir('faces/train/noGlass')
print(len(list_noGlass))

# test_images_file = os.path('./faces/test_images')
# if not test_images_file.exists():
#     os.mkdir('./faces/test_images')

if not os.path.exists('./faces/test_images'):
    os.mkdir('./faces/test_images')
    print('文件夹创建成功！')

for i in range(0, 1000):
    noGlass_imgName = os.path.basename(list_noGlass[i])
    hasGlass_imgName = os.path.basename(list_hasGlass[i])
    if os.path.splitext(noGlass_imgName)[1] != ".jpg":
        continue
    if os.path.splitext(hasGlass_imgName)[1] != ".jpg":
        continue

    shutil.move('./faces/noGlass/'+noGlass_imgName, './faces/test_images/'+noGlass_imgName)
    shutil.move('./faces/hasGlass/'+hasGlass_imgName, './faces/test_images/'+hasGlass_imgName)

