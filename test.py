import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_dataset
from torchvision import transforms
trainset, valset = load_dataset(train=True)

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

# dataiter = iter(trainset)
# images, targets = next(dataiter)

# img_show(images,False)
# img_show(targets,True)
# test_label_transform = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.ToTensor(),
#     ]
# )
# print(type(test_label_transform))
from dataset import domain_generization

original_image, _ = trainset.__getitem__(0)
original_image = original_image / 2 + 0.5
original_image = original_image.numpy()
plt.figure
plt.imshow(np.transpose(original_image, (1, 2, 0)))
plt.show()
scaling_factor = 0.1 # 替换低频区域所占大小
ratio = 1#替换区域中目标域图片的幅度比重 
num_generalized=1

dg_outputs, dg_fre_outputs= np.array(domain_generization(original_image,scaling_factor, ratio,num_generalized,'domain2'))
# print(dg_outputs.shape)
plt.figure
plt.imshow(np.transpose(np.real(dg_outputs[0]), (1, 2, 0)))
plt.show()
print(np.sum(np.abs(dg_outputs - original_image)))