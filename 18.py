import cv2
import os


def save_img():
    video_path = 'E:\DataSets\Video'
    videos = os.listdir(video_path)
    for video_name in videos:
        if not os.path.isfile(video_path + '/' + video_name):
            continue

        file_name = video_name.split('.')[0]
        # if os.path.exists(video_path + '/' + file_name):
        #     continue

        folder_name = video_path + '\\' + file_name
        os.makedirs(folder_name, exist_ok=True)
        vc = cv2.VideoCapture(video_path + '\\' + video_name)  # 读入视频文件
        c = 0
        rval = vc.isOpened()

        while rval:  # 循环读取视频帧
            c = c + 1
            rval, frame = vc.read()
            pic_path = folder_name + '/'
            if rval:
                cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                cv2.waitKey(1)
            else:
                break
        vc.release()
        print('save_success')
        print(folder_name)


if __name__ == '__main__':
    save_img()
