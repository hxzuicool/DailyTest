import os
import shutil

train_list = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\test')

print(len(train_list))

if __name__ == '__main__':

    for i in range(0, len(train_list)):
        dir = os.path.basename(train_list[i])

        # if os.path.exists(r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + '\\spoof'):
        #     list_spoofing = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + '\\spoof')
        #     for k in range(0, len(list_spoofing)):
        #         imgName = os.path.basename(list_spoofing[k])
        #         if (os.path.splitext(imgName)[1] != '.jpg') and (os.path.splitext(imgName)[1] != '.png'):
        #             continue
        #
        #         oldName = r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + '\\spoof\\' + imgName
        #         newName = r'E:\DataSets\CelebA_Spoof\New_Data\new_test\spoof\\' + imgName
        #
        #         shutil.copyfile(oldName, newName)
        #         print(newName)

        if os.path.exists(r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + '\\live'):
            list_living = os.listdir(r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + '\\live')
            for j in range(0, len(list_living)):
                imgName = os.path.basename(list_living[j])
                if os.path.splitext(imgName)[1] != '.jpg' and os.path.splitext(imgName)[1] != '.png':
                    continue

                oldName = r'E:\DataSets\CelebA_Spoof\Data\test\\' + dir + '\\live\\' + imgName
                newName = r'E:\DataSets\CelebA_Spoof\New_Data\new_test\live\\' + imgName

                shutil.copyfile(oldName, newName)
                print(newName)
