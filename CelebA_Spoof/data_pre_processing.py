import os
import shutil

train_list = os.listdir('E:\DataSets\CelebA_Spoof\Data\\train')

print(len(train_list))

if __name__ == '__main__':

    for i in range(0, len(train_list)):
        dir = os.path.basename(train_list[i])

        if os.path.exists('E:\DataSets\CelebA_Spoof\Data\\train\\' + dir + '\spoof'):
            list_spoofing = os.listdir('E:\DataSets\CelebA_Spoof\Data\\train\\' + dir + '\spoof')
            for k in range(0, len(list_spoofing)):
                imgName = os.path.basename(list_spoofing[k])
                if os.path.splitext(imgName)[1] != '.jpg':
                    continue

                oldName = 'E:\DataSets\CelebA_Spoof\Data\\train\\' + dir + '\spoof\\' + imgName
                newName = 'E:\DataSets\CelebA_Spoof\Data\\new_train\spoof\\' + imgName

                shutil.copyfile(oldName, newName)
                print(newName)

        if os.path.exists('E:\DataSets\CelebA_Spoof\Data\\train\\' + dir + '\live'):
            list_living = os.listdir('E:\DataSets\CelebA_Spoof\Data\\train\\' + dir + '\live')
            for j in range(0, len(list_living)):
                imgName = os.path.basename(list_living[j])
                if os.path.splitext(imgName)[1] != '.jpg':
                    continue

                oldName = 'E:\DataSets\CelebA_Spoof\Data\\train\\' + dir + '\live\\' + imgName
                newName = 'E:\DataSets\CelebA_Spoof\Data\\new_train\live\\' + imgName

                shutil.copyfile(oldName, newName)
                print(newName)

