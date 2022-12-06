import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from dataset import load_dataset
from torchvision import transforms

test_set = load_dataset(train=False)

# b = trainset.__getitem__(1)

# temp_shape = 0
# for i in iter(trainset.x):
#     print(i)
#     temp_image = cv2.imread(i)
#     # temp_image = cv2.resize(temp_image,(1634, 1634))
#     print(np.min(temp_image))
    # print(type(temp_image))

    # if not temp_image.shape == temp_shape:
    #     temp_shape = temp_image.shape
    #     print(temp_shape)
# input_pattern = glob.glob(
#     "data/segmentation/Pro1-SegmentationData/Training_data/data/*.bmp"
# )
# # print(input_pattern)   
# input_pattern.sort()
# for i in range(len(input_pattern)):
#     inputpath = input_pattern[i]

#     path = inputpath
#     name = os.path.split(path)[1]
#     name = os.path.splitext(name)[0]
#     # print(name)
#     if os.path.exists(inputpath):
#         print(name)



def img_show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

test = 1
dataiter = iter(test_set)
images, targets,path = next(dataiter)
# print(images)
# img_show(images)

# img_show(targets,True)
# test_label_transform = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.ToTensor(),
#     ]
# )
# print(type(test_label_transform))
from PIL import Image

im = Image.open('./data/Pro1-SegmentationData/Domain3/data/V0001.bmp')
im_l = Image.open('./data/Pro1-SegmentationData/Domain3/label/V0001.bmp')
# plt.imshow((im))
# plt.show()
p1 = random.randint(0,1)
p2 = random.randint(0,1)
im_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(180,resample=False,expand=False),
    transforms.RandomAffine(degrees = 0,translate= (0.1,0.1),fillcolor=0),
    transforms.ColorJitter(brightness=0.5,contrast=0.5,hue = 0.5),
    # transforms.RandomCrop(256),
    transforms.Resize((512,512))
])

im_aug_2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(180,resample=False,expand=False),
    transforms.RandomAffine(degrees = 0,translate= (0.1,0.1),fillcolor=255),
    # transforms.RandomAffine(degrees = 50),
    # transforms.RandomCrop(256),
    transforms.Resize((512,512)),
    transforms.Grayscale(1)
])

# seed = np.random.randint(5000000)
# Observation: after set the same seed value to torch, the image transformed remain the same
# Observation2: the randomness applied to same layer
seed = 50
import torch
# for i in range(2):
#     torch.manual_seed(seed)
#     f = plt.imshow(im_aug(im))
#     plt.show()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# random.seed(seed)
# f = plt.imshow(im_aug(im))
# plt.show()

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# random.seed(seed)
# print(im_l.)
f2 = plt.imshow(im_aug_2(im_l))
plt.show()


# for i in range(5):
#     random.seed(seed)
#     f = plt.imshow(im_aug(im))
#     plt.show()

# from dataset import domain_generization

# original_image, _ = trainset.__getitem__(0)
# original_image = original_image / 2 + 0.5
# original_image = original_image.numpy()
# plt.figure
# plt.imshow(np.transpose(original_image, (1, 2, 0)))
# plt.show()
# scaling_factor = 0.1 # 替换低频区域所占大小
# ratio = 1#替换区域中目标域图片的幅度比重 
# num_generalized=1

# dg_outputs, dg_fre_outputs= np.array(domain_generization(original_image,scaling_factor, ratio,num_generalized,'domain2'))
# # print(dg_outputs.shape)
# plt.figure
# plt.imshow(np.transpose(np.real(dg_outputs[0]), (1, 2, 0)))
# plt.show()
# print(np.sum(np.abs(dg_outputs - original_image)))