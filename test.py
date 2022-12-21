import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from dataset import load_dataset
from torchvision import transforms






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

#################Start Test DG###################
# test_set,validset = load_dataset(train=True,is_DG = True)
# def img_show(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# images, labels = test_set.__getitem__(0)
# plt.subplot(1,2,1)
# img = images / 2 + 0.5     # unnormalize
# npimg = img.numpy()
# plt.imshow(np.transpose(npimg,(1,2,0)))
# plt.subplot(1,2,2)
# plt.imshow(labels,cmap='gray')
# plt.show()
# test = 1
#################End Test DG#####################

# #################Start Test iter#################
# import torch
# import torchvision
# test_set,validset = load_dataset(train=True,is_DG = True)
# trainloader = torch.utils.data.DataLoader(test_set, batch_size=4,
#                                           shuffle=True, num_workers=2)
# def img_show(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# dataiter = iter(trainloader)
# images, targets = next(dataiter)

# plt.subplot(1,2,1)
# img = images / 2 + 0.5     # unnormalize
# npimg = img.numpy()
# img_show(torchvision.utils.make_grid(images))
# #################ENd Test iter###################


# dataiter = iter(test_set)

# import torch
# import torchvision
# trainloader = torch.utils.data.DataLoader(test_set, batch_size=4,
#                                           shuffle=True, num_workers=2)

# images, targets = test_set.__getitem__(0)

# dataiter = iter(trainloader)
# images, targets = next(dataiter)
# img_show(images)

# for i, (inputs, target) in enumerate(trainloader):
#     img_show(torchvision.utils.make_grid(inputs))

    # pass
# for i in range(5):
#     images, targets = next(dataiter)
# # print(images)
#     img_show(images)
#     # img_show(targets)
#     plt.imshow(targets,cmap='gray')
#     plt.show()

# img_show(targets,True)
# test_label_transform = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.ToTensor(),
#     ]
# )
# print(type(test_label_transform))
# from PIL import Image

# im = Image.open('./data/Pro1-SegmentationData/Domain3/data/V0001.bmp')
# im_l = Image.open('./data/Pro1-SegmentationData/Domain3/label/V0001.bmp')
# # plt.imshow((im))
# # plt.show()
# p1 = random.randint(0,1)
# p2 = random.randint(0,1)
# im_aug = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     # transforms.RandomRotation(180,resample=False,expand=False),
#     transforms.RandomAffine(degrees = 0,translate= (0.1,0.1),fillcolor=0),
#     transforms.ColorJitter(brightness=0.5,contrast=0.5,hue = 0.5),
#     # transforms.RandomCrop(256),
#     transforms.Resize((512,512))
# ])

# im_aug_2 = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     # transforms.RandomRotation(180,resample=False,expand=False),
#     transforms.RandomAffine(degrees = 0,translate= (0.1,0.1),fillcolor=255),
#     # transforms.RandomAffine(degrees = 50),
#     # transforms.RandomCrop(256),
#     transforms.Resize((512,512)),
#     transforms.Grayscale(1)
# ])

# seed = np.random.randint(5000000)
# Observation: after set the same seed value to torch, the image transformed remain the same
# Observation2: the randomness applied to same layer
seed = 50

# temp_image = np.asarray(im,dtype = np.float64)/255
# # plt.imshow(temp_image)
# # plt.show()
# # img_show(temp_image)
# # print(temp_image.shape)
# # print(temp_image.shape)
# from dataset import domain_generization
# temp_images, _ = domain_generization(np.transpose(temp_image,(2,1,0)))
# print(type(temp_images))
# temp_image = temp_images[0]
# print(temp_image.shape)
# plt.imshow(np.transpose(temp_image, (1, 2, 0)))
# plt.show()
# print(np.max(temp_image))
# print(np.min(temp_image))
# # print(temp_image.shape)
# import torch
# from PIL import Image
# print(temp_image)



# temp_image = Image.fromarray(np.uint8(np.clip(np.transpose(temp_image, (1, 2, 0)),0,1) * 255)).convert('RGB')
# temp_image.show()
# # temp_image = torch.from_numpy(temp_image)
# # temp_image = PIL2Tensor(np.transpose(temp_image, (1, 2, 0)))
# print(type(temp_image))


# import torch
# plt.imshow(np.transpose(temp_image, (1, 2, 0)))
# plt.show()

# import torch
# # for i in range(2):
# #     torch.manual_seed(seed)
# #     f = plt.imshow(im_aug(im))
# #     plt.show()
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# # random.seed(seed)
# img_show(im_aug(im))
# # plt.show()

# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# # random.seed(seed)
# # print(im_l.)
# img_show(im_aug_2(im_l))



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


# ######################Test Dice Loss####################
# import torch
# import torch.nn as nn

# class BinaryDiceLoss(nn.Module):
#     def __init__(self) -> None:
#         super(BinaryDiceLoss, self).__init__()
#     def forward(self,inputs, targets):
#         # Inputs and targets should both be a long type 
#         # Batch numbers
#         N = targets.size()[0]
#         # Smooth parameters
#         smooth = 1
#         inputs_flat = inputs.view(N,-1)
#         print(inputs_flat.size())
#         targets_flat = targets.view(N,-1)

#         intersection = inputs_flat * targets_flat
        
#         dice_eff = (2 * intersection.sum(1) + smooth) / (inputs_flat.sum(1) + targets_flat.sum(1) + smooth)
#         print(intersection.size())
#         loss = 1 - dice_eff.sum()/N
#         return loss 


# criterion = BinaryDiceLoss()

# a = torch.ones(2,3,16,16)
# b = torch.zeros(2,3,16,16)

# loss = criterion(a,b)
# print(loss)



# ######################End Test Dice Loss################

from dataset import load_dataset

a,b = load_dataset()