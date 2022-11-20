import glob
import os
import cv2
import numpy as np

from dataset import load_dataset

trainset, valset = load_dataset(train=True)

# b = trainset.__getitem__(1)

temp_shape = 0
for i in iter(trainset.x):
    print(i)
    temp_image = cv2.imread(i)
    # temp_image = cv2.resize(temp_image,(1634, 1634))
    print(np.min(temp_image))
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