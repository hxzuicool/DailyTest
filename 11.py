import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from PIL.Image import Image
import os

batch_size = 100

train_dataset = dsets.MNIST('./data/pymnist', train=True, transform=transforms.ToTensor, download=True)
test_dataset = dsets.MNIST('./data/pymnist', train=False, transform=transforms.ToTensor, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# CelebA_dataset = dsets.CelebA('./data/CelebA', "train", transform=transforms.ToTensor, download=True)

torch.nn.MSELoss(reduction='mean')

torch.nn.CrossEntropyLoss()


# class myDataSets(dsets):
#     def __init__(self, path, transform=transforms.ToTensor):
#         self.path = 'E:\DataSets\Images_CelebA\img_align_celeba'
#         self.transform = transforms.ToTensor
#         self.images = os.listdir(self.path)
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, index):  # 根据索引index返回图像及标签
#         image_index = self.images[index]  # 根据索引获取图像文件名称
#         img_path = os.path.join(self.path, image_index)  # 获取图像的路径或目录
#         img = Image.open(img_path).convert('RGB')  # 读取图像
#
#         # 根据目录名称获取图像标签（cat或dog）
#         label = img_path.split('\\')[-1].split('.')[0]
#         # 把字符转换为数字cat-0，dog-1
#         label = 1 if 'dog' in label else 0
#
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label
