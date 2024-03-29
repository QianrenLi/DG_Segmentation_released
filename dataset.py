import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageOps
import random

cv2.setNumThreads(1)


def analyze_name(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name


def random_crop(img, gt, size=[0.5, 0.8]):
    """Crop patches in ROI with random size"""
    import random
    img =np.asarray(img,dtype = np.float64)/255
    gt = np.asarray(gt,dtype = np.float64)/255
    roi = gt[:,:,0]
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

    cropped_img = img[iy - ip_l : iy + ip_r, ix - ip_l : ix + ip_r,:]
    cropped_gt = gt[iy - ip_l : iy + ip_r, ix - ip_l : ix + ip_r,:]

    return cropped_img, cropped_gt


class MNIST(Dataset):
    def __init__(self, x, y, names, im_transform, label_transform, train=False,is_DG = 0):
        self.im_transform = im_transform
        self.label_transform = label_transform
        assert len(x) == len(y)
        assert len(x) == len(names)
        self.dataset_size = len(y)
        self.x = x
        self.y = y
        self.names = names
        self.train = train
        self.is_DG = is_DG

    def __len__(self):
        return self.dataset_size

    def _get_index(self, idx):
        return idx

    def __getitem__(self, idx):
        # is_DG: A bool_value to determine whether the DG is applied
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self._get_index(idx)
        # To enable the same transformation, manual seed is applied
        


        seed = np.random.randint(5000000)
        # idx is int number
        # self.x is the training set file name vector, self.y is the label set file name vector
        # e.g [v001.bmp,v002.bmp,...,]
        wh = 1200
        temp_image = Image.open(self.x[idx],mode='r')
        label_image = Image.open(self.y[idx])
        # print(temp_image.size)

        # print(temp_image.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(temp_image)
        # plt.show()
        # Do domain generization
        if self.is_DG != 0:
            # Change into numpy
            temp_image_t = np.asarray(temp_image,dtype = np.float64)/255

            # Resize the DG image
            temp_image_t = cv2.resize(temp_image_t,(32 * 12, 32 * 12))
            temp_image_t = np.transpose(temp_image_t, (2,0,1))
            temp_images,_ = domain_generization(temp_image_t,domain=self.is_DG)
            temp_image_t = np.real(temp_images[0])
            
            # Change into PIL (H W C)
            temp_image = Image.fromarray(np.uint8(np.clip(np.transpose(temp_image_t, (1, 2, 0)),0,1) * 255))
            # temp_image.show()

        # if self.train == True:
        #     temp_image,label_image = random_crop(temp_image,label_image)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        temp_image = self.im_transform(temp_image)
        # print(type(temp_image))
        # print(temp_image.dtype)
        # seed = np.random.randint(5000000)

        # print(type(temp_image))
        # Image size normalized -- image size is different 1634, 1634
        # N = 32 * 50
        # temp_image = np.array(cv2.resize(temp_image,(N, N)),dtype="uint8")

        # Transpose dimension from H * W * C to C * H * W 
        # temp_image = temp_image.transpose(2,0,1)

        # Return the label image:
        # print(self.y)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        label_image = self.label_transform(label_image)
        # if self.is_DG:
        #     label_image = np.asarray(label_image)
        #     label_image = Image.fromarray(label_image)
        label_image = label_image.squeeze(dim = 0)
        
        label_image =  torch.cat((1-label_image.unsqueeze(0), (label_image).unsqueeze(0)),0)
        
        # label_image = 1 - label_image 
        # print(label_image.shape)
        # print(label_image)
        # label_image = np.array((cv2.resize(label_image,(N, N))/255),dtype="uint8")
        # label_image = label_image.transpose(1,0)

        # Return image, label file name, file name
        if self.train == True:
            return temp_image,label_image
        return temp_image,label_image,self.names[idx]


def load_name(train_data_str = "./data/Pro1-SegmentationData/Training_data/data/*.bmp", \
                train_label_str = "./data/Pro1-SegmentationData/Training_data/label/{}.bmp", \
                valid_data_str = "./data/Pro1-SegmentationData/Training_data/data/*.bmp", \
                valid_label_str = "./data/Pro1-SegmentationData/Training_data/label/{}.bmp", \
                test_data_str = "./data/Pro1-SegmentationData/Domain2/data/*.bmp", \
                test_label_str = "./data/Pro1-SegmentationData/Domain2/label/{}.bmp",div_factor = 5,fold_id = 4):

    inputs, targets, names = [], [], []
    val_inputs, val_targets, val_names = [], [], []
    test_inputs, test_targets, test_names = [], [], []

    # This link represents the file link
    input_pattern = glob.glob(
        train_data_str
    )
    targetlist = (
        train_label_str
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

    # val_input_pattern = glob.glob(
    #     valid_data_str
    # )
    # val_targetlist = valid_label_str

    # val_input_pattern.sort()

    # for j in tqdm(range(len(val_input_pattern))):
    #     val_inputpath = val_input_pattern[j]
    #     val_name = analyze_name(val_inputpath)
    #     val_targetpath = val_targetlist.format(str(val_name))

    #     if os.path.exists(val_inputpath):
    #         val_inputs.append(val_inputpath)
    #         val_targets.append(val_targetpath)
    #         val_names.append(val_name)
    
    data_size = len(inputs)
    div_factor = div_factor
    segments_length = int(data_size/div_factor)
    slice_pos = fold_id
    val_inputs = inputs[slice_pos * segments_length: (slice_pos +1) *segments_length ]
    val_targets = targets[slice_pos * segments_length: (slice_pos +1) *segments_length ]
    val_names = names[slice_pos * segments_length: (slice_pos +1) *segments_length ]

  
    inputs = np.concatenate((inputs[0:slice_pos * segments_length],inputs[(slice_pos +1) *segments_length :]),axis= 0) 
    targets = np.concatenate((targets[0:slice_pos * segments_length],targets[(slice_pos +1) *segments_length :]),axis= 0) 
    names = np.concatenate((names[0:slice_pos * segments_length],names[(slice_pos +1) *segments_length :]),axis= 0) 


    test_input_pattern = glob.glob(
        test_data_str
    )
    test_targetlist = (
        test_label_str
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


def load_dataset(train=True,is_vert_flip = True,is_rotate = True,is_translate = True,is_color_jitter = True,is_random_crop = True,is_DG = False, \
                train_data_str = "./data/Pro1-SegmentationData/Training_data/data/*.bmp", \
                train_label_str = "./data/Pro1-SegmentationData/Training_data/label/{}.bmp", \
                valid_data_str = "./data/Pro1-SegmentationData/Training_data/data/*.bmp", \
                valid_label_str = "./data/Pro1-SegmentationData/Training_data/label/{}.bmp", \
                test_data_str = "./data/Pro1-SegmentationData/Domain2/data/*.bmp", \
                test_label_str = "./data/Pro1-SegmentationData/Domain2/label/{}.bmp",div_factor = 5, fold_id = 4):
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
    ) = load_name(train_data_str,train_label_str,valid_data_str,valid_label_str,test_data_str,test_label_str,div_factor,fold_id)

    # print("Length of new inputs:", len(inputs))
    # mean & variance
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # normalize = transforms.Normalize(mean=[0.26005924, 0.15255839, 0.075675726], std=[5, 5, 5])
    # normalize = transforms.Normalize(mean = [0.4227116, 0.2817605, 0.22160497], std = [0.289763, 0.2044466, 0.157756])
    # normalize = transforms.Normalize(mean = [0.26005924, 0.15255839, 0.075675726], std=[0.17948404, 0.10386878, 0.053291164])

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

    # Transformation setting
    input_transform_list = []
    label_transform_list = []
    input_transform_list.append(transforms.ToTensor())
    label_transform_list.append(transforms.ToTensor())

    # Notice the location of eye are generally in [40: 1500] [0: 400] 
    # if is_random_crop:
    #     input_transform_list.append(transforms.RandomCrop(600))
    #     label_transform_list.append(transforms.RandomCrop(600))

    input_transform_list.append(transforms.Resize([32 * 12,32 * 12]))
    label_transform_list.append(transforms.Resize([32 * 12, 32 * 12]))
    if is_rotate:
        input_transform_list.append(transforms.RandomRotation(30,expand=False,fill=0))
        label_transform_list.append(transforms.RandomRotation(30,expand=False,fill=1))
    if is_translate:
        # 0.1,0.1 is the factor
        input_transform_list.append(transforms.RandomAffine(degrees = 0,translate= (0.1,0.1),fill=0))
        label_transform_list.append(transforms.RandomAffine(degrees = 0,translate= (0.1,0.1),fill=1))
    if is_vert_flip:
        input_transform_list.append(transforms.RandomVerticalFlip())
        label_transform_list.append(transforms.RandomVerticalFlip())
        input_transform_list.append(transforms.RandomHorizontalFlip())
        label_transform_list.append(transforms.RandomHorizontalFlip())
    if is_color_jitter:
        input_transform_list.append(transforms.ColorJitter(brightness=0.3,contrast=0.3, saturation= 0.3))
    
        

    input_transform_list.append(transforms.Resize([32 * 12, 32 * 12]))
    label_transform_list.append(transforms.Resize([32 * 12, 32 * 12]))

    label_transform_list.append(transforms.Grayscale(1))
    input_transform_list.append(normalize)



    transform = transforms.Compose(
        input_transform_list
    )
    label_transform = transforms.Compose(
        label_transform_list
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([32*12,32*12]),
            normalize,
        ]
    )
    test_label_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(1),
            transforms.Resize([32*12,32*12]),
        ]
    )

    train_dataset = MNIST(
        X_train,
        y_train,
        names_train,
        im_transform=transform,
        label_transform=label_transform,
        train=True,
        is_DG = is_DG,
    )
    val_dataset = MNIST(
        X_val,
        y_val,
        names_val,
        im_transform=test_transform,
        label_transform=test_label_transform,
        train=False,
        is_DG = False,
    )
    test_dataset = MNIST(
        X_test,
        y_test,
        names_test,
        im_transform=test_transform,
        label_transform=test_label_transform,
        train=False,
        is_DG = False,
    )

    if train:
        return train_dataset, val_dataset
    else:
        return test_dataset

def domain_generization(original_image, scaling_factor = 0.03, ratio = 1, num_generalized = 1,domain = 4,is_Amplitude = True,is_return_Domain = False):
    # Requiring unnormalized input image shape as (C,H,W)
    # domain: 'domain1' = 1, 'domain2' = 2,'domain3' = 3,'random' = 4
    # Return C*H*W images and log normalized fftshit frequency.


    domain_pattern_1 = glob.glob(
        "./data/Pro1-SegmentationData/Domain1/data/*.jpg"
    )
    domain_pattern_2 = glob.glob(
        "./data/Pro1-SegmentationData/Domain2/data/*.bmp"
    )
    domain_pattern_3 = glob.glob(
        "./data/Pro1-SegmentationData/Domain3/data/*.bmp"
    )

    domain_pattern_1.sort()
    domain_pattern_2.sort()
    domain_pattern_3.sort()
    inputs = []
    # Here the domain set contain all the data

    if domain == 4 or domain == 1:
        for i in range(len(domain_pattern_1)):
            inputpath = domain_pattern_1[i]
            if os.path.exists(inputpath):
                inputs.append(inputpath)

    if domain == 4 or domain == 2:
        for i in range(len(domain_pattern_2)):
            inputpath = domain_pattern_2[i]
            if os.path.exists(inputpath):
                inputs.append(inputpath)

    if domain == 4 or domain == 3:
        for i in range(len(domain_pattern_3)):
            inputpath = domain_pattern_3[i]
            if os.path.exists(inputpath):
                inputs.append(inputpath)

    inputs = np.array(inputs)
    length_inputs = len(inputs)

    H_value = original_image.shape[1]
    W_value = original_image.shape[2]


    H_left = np.ceil(H_value/2 - H_value * scaling_factor/2).astype(int)
    H_right = np.ceil(H_value/2 + H_value * scaling_factor/2+1).astype(int)
    W_left = np.ceil(W_value/2 - W_value * scaling_factor/2).astype(int)
    W_right = np.ceil(W_value/2 + W_value * scaling_factor/2+1).astype(int)

    indexs = random.sample(range(length_inputs),num_generalized)
    dg_outputs = np.zeros((num_generalized,3,H_value,W_value))
    dg_fre_outputs =np.zeros((num_generalized,3,H_value,W_value),dtype= complex)

    #Image denormalized
    

    for i in range(num_generalized):
        # print(type(str(inputs[indexs[i]])))
        generalized_image = cv2.imread((inputs[indexs[i]]))
        generalized_image = cv2.cvtColor(generalized_image,cv2.COLOR_BGR2RGB)
        
        generalized_image = np.array(cv2.resize(generalized_image,(H_value, W_value)),dtype="uint8")
        generalized_image = generalized_image.transpose(2,0,1)
        generalized_image = np.asarray(generalized_image, np.float64) / 255

        if is_return_Domain:
            generalized_image_dump = generalized_image
        # print(np.max(generalized_image))
        # generalized_image = np.array(cv2.resize(generalized_image,(H_value, W_value)),dtype="uint8")
        # generalized_image = generalized_image.transpose(2,0,1)
        
        # Do FFT to each channel
        generalized_image_frequency_domain = np.fft.fftshift(np.fft.fft2(generalized_image,axes=(-2, -1)),axes=(-2, -1))
        original_image_frequency_domain = np.fft.fftshift(np.fft.fft2(original_image,axes=(-2, -1)),axes=(-2, -1))

        amplitude_generalized_image = np.abs(generalized_image_frequency_domain)
        amplitude_original_image= np.abs(original_image_frequency_domain)
        phase_generalized_image  = np.angle(generalized_image_frequency_domain)
        phase_original_image = np.angle(original_image_frequency_domain)
        

        power_generelized = np.linalg.norm(amplitude_generalized_image[:,H_left:H_right,W_left:W_right])
        power_original = np.linalg.norm(amplitude_original_image[:,H_left:H_right,W_left:W_right])

        if is_Amplitude == True:
        # Replace the amplitude
            amplitude_original_image[:,H_left:H_right,W_left:W_right] \
                = (1-ratio)* amplitude_original_image[:,H_left:H_right,W_left:W_right] \
                + ratio* amplitude_generalized_image[:,H_left:H_right,W_left:W_right] * power_original/power_generelized
            # amplitude_original_image[:,H_left:H_right,W_left:W_right] = amplitude_original_image[:,H_left:H_right,W_left:W_right] - amplitude_original_image[:,H_left:H_right,W_left:W_right]
            # Output generalized image
            generalized_freq = amplitude_original_image * np.exp(1j*phase_original_image)
            generalized_image = np.real(np.fft.ifft2(np.fft.fftshift(generalized_freq,axes=(2,1)),axes=(-2, -1)))
        else:
            phase_original_image[:,H_left:H_right,W_left:W_right] \
                = (1-ratio)* phase_original_image[:,H_left:H_right,W_left:W_right] \
                + ratio* phase_generalized_image[:,H_left:H_right,W_left:W_right] 
            # amplitude_original_image[:,H_left:H_right,W_left:W_right] = amplitude_original_image[:,H_left:H_right,W_left:W_right] - amplitude_original_image[:,H_left:H_right,W_left:W_right]
            # Output generalized image
            generalized_freq = amplitude_original_image * np.exp(1j*phase_original_image)
            generalized_image = np.real(np.fft.ifft2(np.fft.fftshift(generalized_freq,axes=(2,1)),axes=(-2, -1)))

        # print(generalized_image.shape)
        # print(type(generalized_image))
        dg_outputs[i] = generalized_image
        # print(generalized_freq.shape)
        dg_fre_outputs[i] = generalized_freq
    
    if is_return_Domain:
        return dg_outputs,dg_fre_outputs,generalized_image_dump
    else:
        return dg_outputs,dg_fre_outputs