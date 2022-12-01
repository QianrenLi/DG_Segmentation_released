import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

cv2.setNumThreads(1)


def analyze_name(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name


def random_crop(img, gt, roi, size=[0.2, 0.8]):
    """Crop patches in ROI with random size"""
    import random

    ih, iw = img.shape[:2]
    ip = random.randrange(int(ih * size[0]), int(ih * size[1]))
    ip_l = ip // 2
    ip_r = ip - ip_l

    ix = random.randrange(ip_l, iw - ip_r + 1)
    iy = random.randrange(ip_l, iw - ip_r + 1)
    if roi is not None:
        # crop a patch in the region of interest
        while roi[iy, ix] == 0:
            ix = random.randrange(ip_l, iw - ip_r + 1)
            iy = random.randrange(ip_l, iw - ip_r + 1)

    cropped_img = img[iy - ip_l : iy + ip_r, ix - ip_l : ix + ip_r]
    cropped_gt = gt[iy - ip_l : iy + ip_r, ix - ip_l : ix + ip_r]

    return cropped_img, cropped_gt


class MNIST(Dataset):
    def __init__(self, x, y, names, im_transform, label_transform, train=False):
        self.im_transform = im_transform
        self.label_transform = label_transform
        assert len(x) == len(y)
        assert len(x) == len(names)
        self.dataset_size = len(y)
        self.x = x
        self.y = y
        self.names = names
        self.train = train

    def __len__(self):
        return self.dataset_size

    def _get_index(self, idx):
        return idx

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self._get_index(idx)

        # idx is int number
        # self.x is the training set file name vector, self.y is the label set file name vector
        # e.g [v001.bmp,v002.bmp,...,]
        temp_image = cv2.imread(self.x[idx])

        # Image size normalized -- image size is different 1634, 1634
        N = 32 * 12
        temp_image = np.array(cv2.resize(temp_image,(N, N)),dtype="uint8")

        # Transpose dimension from H * W * C to C * H * W 
        temp_image = temp_image.transpose(2,0,1)

        # Return the label image:
        label_image = cv2.imread(self.y[idx],cv2.IMREAD_GRAYSCALE)
        label_image = np.array((cv2.resize(label_image,(N, N))/255),dtype="uint8")
        # label_image = label_image.transpose(1,0)
        # Return image, label file name, file name
        if self.train == True:
            return temp_image,label_image
        return temp_image,label_image,self.names[idx]


def load_name():

    inputs, targets, names = [], [], []
    val_inputs, val_targets, val_names = [], [], []
    test_inputs, test_targets, test_names = [], [], []

    # This link represents the file link
    input_pattern = glob.glob(
        "./data/Pro1-SegmentationData/Training_data/data/*.bmp"
    )
    targetlist = (
        "./data/Pro1-SegmentationData/Training_data/label/{}.bmp"
    )

    input_pattern.sort()
    
    for i in tqdm(range(len(input_pattern))):
        inputpath = input_pattern[i]
        name = analyze_name(inputpath)
        targetpath = targetlist.format(str(name))

        if os.path.exists(inputpath):
            inputs.append(inputpath)
            targets.append(targetpath)
            names.append(name)

    inputs = np.array(inputs)
    targets = np.array(targets)
    names = np.array(names)

    val_input_pattern = glob.glob(
        "data/Pro1-SegmentationData/Training_data/data/*.bmp"
    )
    val_targetlist = "data/Pro1-SegmentationData/Training_data/label/{}.bmp"

    val_input_pattern.sort()

    for j in tqdm(range(len(val_input_pattern))):
        val_inputpath = val_input_pattern[j]
        val_name = analyze_name(val_inputpath)
        val_targetpath = val_targetlist.format(str(val_name))

        if os.path.exists(val_inputpath):
            val_inputs.append(val_inputpath)
            val_targets.append(val_targetpath)
            val_names.append(val_name)

    test_input_pattern = glob.glob(
        "data/Pro1-SegmentationData/Domain2/data/*.bmp"
    )
    test_targetlist = (
        "data/Pro1-SegmentationData/Domain2/label/{}.bmp"
    )

    test_input_pattern.sort()

    for j in tqdm(range(len(test_input_pattern))):
        test_inputpath = test_input_pattern[j]
        test_name = analyze_name(test_inputpath)
        test_targetpath = test_targetlist.format(str(test_name))

        if os.path.exists(test_inputpath):
            test_inputs.append(test_inputpath)
            test_targets.append(test_targetpath)
            test_names.append(test_name)

    test_inputs = np.array(test_inputs)
    test_targets = np.array(test_targets)
    test_names = np.array(test_names)

    assert len(inputs) == len(targets)
    assert len(targets) == len(names)

    return (
        inputs,
        targets,
        names,
        val_inputs,
        val_targets,
        val_names,
        test_inputs,
        test_targets,
        test_names,
    )


def load_dataset(train=True):
    (
        inputs,
        targets,
        names,
        val_inputs,
        val_targets,
        val_names,
        test_inputs,
        test_targets,
        test_names,
    ) = load_name()
    print("Length of new inputs:", len(inputs))
    # mean & variance
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    X_trainset, X_test = inputs, test_inputs
    y_trainset, y_test = targets, test_targets
    train_names_set, names_test = names, test_names

    X_train, X_val, y_train, y_val, names_train, names_val = (
        X_trainset,
        val_inputs,
        y_trainset,
        val_targets,
        train_names_set,
        val_names,
    )

    # transform input images and construct datasets
    transform = transforms.Compose(
        [
            # Try some transformation
            transforms.ToTensor(),
            normalize,
        ]
    )
    label_transform = transforms.Compose(
        [
            # Try some transformation
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_label_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
        ]
    )

    train_dataset = MNIST(
        X_train,
        y_train,
        names_train,
        im_transform=transform,
        label_transform=label_transform,
        train=True,
    )
    val_dataset = MNIST(
        X_val,
        y_val,
        names_val,
        im_transform=test_transform,
        label_transform=test_label_transform,
        train=False,
    )
    test_dataset = MNIST(
        X_test,
        y_test,
        names_test,
        im_transform=test_transform,
        label_transform=test_label_transform,
        train=False,
    )

    if train:
        return train_dataset, val_dataset
    else:
        return test_dataset
