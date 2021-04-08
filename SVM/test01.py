import os
import numpy as np
from skimage import feature as skif
from skimage import io, transform
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
import cv2
import torch
import torchvision
from torchvision import datasets

# ImageFile.LOAD_TRUNCATED_IMAGES = True

__file__ = r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random'
# 全局变量
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')  # images的绝对路径
LIVE_IMAGE_DIR = os.path.join(IMAGES_DIR, 'live')
SPOOF_IMAGE_DIR = os.path.join(IMAGES_DIR, 'spoof')
RESIZE_LIVE_IMAGE_DIR = os.path.join(IMAGES_DIR, 'resize_live')
RESIZE_SPOOF_IMAGE_DIR = os.path.join(IMAGES_DIR, 'resize_spoof')
IMG_TYPE = 'jpg'  # 图片类型
IMG_WIDTH = 256
IMG_HEIGHT = 256


# def resize_image(file_in, file_out, width, height):
#     img = io.imread(file_in)
#     out = transform.resize(img, (width, height),
#                            mode='reflect')  # mode {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
#     io.imsave(file_out, out)
def resize_image(filepath, filename, height_dst, width_dst, isLive='live'):
    img = cv2.imread(filepath)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    fx = width_dst / width
    fy = height_dst / height
    resize_img = cv2.resize(img, (0, 0), fx=fx, fy=fy)

    cv2.imwrite(r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize\\' + isLive + '\\' + filename, resize_img)
    return resize_img


# def load_images(images_list, width, height):
#     data = np.zeros((len(images_list), width, height))  # 创建多维数组存放图片
#     for index, image in enumerate(images_list):
#         # image_data = io.imread(image, as_gray=True)
#         image_data = io.imread(image)
#         # data[index][:, :, :] = image_data  # 读取图片存进numpy数组
#         bs = list()
#         bs.append(image_data[np.newaxis, :])
#         data = np.concatenate(bs, axis=0)
#     print(data.shape)
#     return data
def load_images(images_list, width, height):
    data = np.zeros((len(images_list), width, height, 3))  # 创建多维数组存放图片(1000, 256, 256, 3)
    for index, image in enumerate(images_list):
        image_data = io.imread(image)
        data[index, :, :, :] = image_data  # 读取图片存进numpy数组
    return data


def split_data(file_path_list, lables_list, rate=0.5):
    if rate == 1.0:
        return file_path_list, lables_list, file_path_list, lables_list
    list_size = len(file_path_list)
    train_list_size = int(list_size * rate)
    selected_indexes = random.sample(range(list_size), train_list_size)
    train_file_list = []
    train_label_list = []
    test_file_list = []
    test_label_list = []
    for i in range(list_size):
        if i in selected_indexes:
            train_file_list.append(file_path_list[i])
            train_label_list.append(lables_list[i])
        else:
            test_file_list.append(file_path_list[i])
            test_label_list.append(lables_list[i])
    return train_file_list, train_label_list, test_file_list, test_label_list


def get_lbp_data(images_data_list, hist_size=256, lbp_radius=1, lbp_point=8):
    n_images = images_data_list.shape[0]
    print(images_data_list.shape)
    hist = np.zeros((n_images, hist_size))
    for i in np.arange(n_images):
        image_data = images_data_list[i, :, :, 0]
        # 使用LBP方法提取图像的纹理特征.
        lbp = skif.local_binary_pattern(image_data, lbp_point, lbp_radius, 'default')
        # 统计图像的直方图
        max_bins = int(lbp.max() + 1)
        # hist size:256
        hist[i], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))

    return hist


def main():
    # 获取图片列表
    live_list = os.listdir(r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random\live')
    spoof_list = os.listdir(r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random\spoof')
    # 调整图片大小
    # for index, picName in enumerate(live_list):
    #     resize_image(r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random\live\\' + picName, picName, IMG_WIDTH, IMG_HEIGHT, isLive='live')
    # for index, picName in enumerate(spoof_list):
    #     resize_image(r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random\spoof\\' + picName, picName, IMG_WIDTH, IMG_HEIGHT, isLive='spoof')
    # 调整后的图片列表
    live_filename = os.listdir(r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize\live')
    live_file_path_list = live_filename
    for f in range(len(live_filename)):
        live_file_path_list[f] = r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize\live\\' + live_filename[
            f]
    spoof_filename = os.listdir(r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize\spoof')
    spoof_file_path_list = spoof_filename
    for f in range(len(live_filename)):
        spoof_file_path_list[f] = r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize\spoof\\' + \
                                  spoof_filename[f]
    # 切分数据集
    train_file_list1, train_label_list1, test_file_list1, test_label_list1 = split_data(live_file_path_list,
                                                                                        [1] * len(live_file_path_list),
                                                                                        rate=0.5)
    train_file_list0, train_label_list0, test_file_list0, test_label_list0 = split_data(spoof_file_path_list,
                                                                                        [-1] * len(
                                                                                            spoof_file_path_list),
                                                                                        rate=0.5)
    # 合并数据集
    train_file_list = train_file_list0 + train_file_list1
    train_label_list = train_label_list0 + train_label_list1
    test_file_list = test_file_list0 + test_file_list1
    test_label_list = test_label_list0 + test_label_list1
    # 载入图片
    train_image_array = load_images(train_file_list, width=IMG_WIDTH, height=IMG_HEIGHT)
    train_label_array = np.array(train_label_list)
    test_image_array = load_images(test_file_list, width=IMG_WIDTH, height=IMG_HEIGHT)
    test_label_array = np.array(test_label_list)

    # trainset = datasets.ImageFolder(r'E:\DataSets\CelebA_Spoof\New_Data\face_region_random_resize', transform=torchvision.transforms.ToPILImage())
    # dataloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    #
    # for batch_idx, (images, labels) in enumerate(dataloader):
    #     print(batch_idx)
    #     print(images, labels)

    # print(train_image_array.shape, train_label_array.shape)
    # 获取LBP特征
    train_hist_array = get_lbp_data(train_image_array, hist_size=256, lbp_radius=1, lbp_point=8)
    test_hist_array = get_lbp_data(test_image_array, hist_size=256, lbp_radius=1, lbp_point=8)
    # 选取svm里面的SVR作为训练模型
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # SVC, NuSVC, SVR, NuSVR, OneClassSVM, LinearSVC, LinearSVR
    # 训练和测试
    score = OneVsRestClassifier(svr_rbf, n_jobs=-1).fit(train_hist_array, train_label_array).score(test_hist_array,
                                                                                                   test_label_array)  # n_jobs是cpu数量, -1代表所有
    print(svr_rbf)
    print(score)
    return score


if __name__ == '__main__':
    n = 10
    scores = []
    for i in range(n):
        s = main()
        scores.append(s)
    max_s = max(scores)
    min_s = min(scores)
    avg_s = sum(scores) / float(n)
    print('==========\nmax: %s\nmin: %s\navg: %s' % (max_s, min_s, avg_s))
