from dataset import load_dataset
from dataset import domain_generization
import dis_rep as disp  
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Do DG to each figure
def domain_generization(original_image,inputs,distance_metric, scaling_factor = 0.5, ratio = 0.5):
    inputs = np.array(inputs)
    length_inputs = len(inputs)

    H_value = original_image.shape[1]
    W_value = original_image.shape[2]


    H_left = np.ceil(H_value/2 - H_value * scaling_factor/2).astype(int)
    H_right = np.ceil(H_value/2 + H_value * scaling_factor/2).astype(int)
    W_left = np.ceil(W_value/2 - W_value * scaling_factor/2).astype(int)
    W_right = np.ceil(W_value/2 + W_value * scaling_factor/2).astype(int)

    dg_outputs = np.zeros((3,H_value,W_value))
    dg_fre_outputs =np.zeros((3,H_value,W_value),dtype= complex)

    PSNR_list= []
    channel_PSNR_list = []

    for i in range(len(inputs)):
        # print(type(str(inputs[indexs[i]])))
        generalized_image = cv2.imread((inputs[i]))
        generalized_image = cv2.cvtColor(generalized_image,cv2.COLOR_BGR2RGB)
        
        generalized_image = np.array(cv2.resize(generalized_image,(H_value, W_value)),dtype="uint8")
        generalized_image = generalized_image.transpose(2,0,1)
        generalized_image = np.asarray(generalized_image, np.float64) / 255
        # print(np.max(generalized_image))
        # generalized_image = np.array(cv2.resize(generalized_image,(H_value, W_value)),dtype="uint8")
        # generalized_image = generalized_image.transpose(2,0,1)
        
        # Do FFT to each channel
        amplitude_generalized_image = np.abs(np.fft.fftshift(np.fft.fft2(generalized_image,axes=(-2, -1)),axes=(-2, -1)))
        amplitude_original_image= np.abs(np.fft.fftshift(np.fft.fft2(original_image,axes=(-2, -1)),axes=(-2, -1)))
        phase_original_image = np.angle(np.fft.fftshift(np.fft.fft2(original_image,axes=(-2, -1)),axes=(-2, -1)))

        power_generelized = np.linalg.norm(amplitude_generalized_image[:,H_left:H_right,W_left:W_right])
        power_original = np.linalg.norm(amplitude_original_image[:,H_left:H_right,W_left:W_right])

        # Replace the amplitude
        amplitude_original_image[:,H_left:H_right,W_left:W_right] \
            = (1-ratio)* amplitude_original_image[:,H_left:H_right,W_left:W_right] \
            + ratio* amplitude_generalized_image[:,H_left:H_right,W_left:W_right] * power_original/power_generelized
        # amplitude_original_image[:,H_left:H_right,W_left:W_right] = amplitude_original_image[:,H_left:H_right,W_left:W_right] - amplitude_original_image[:,H_left:H_right,W_left:W_right]
        # Output generalized image
        generalized_freq = amplitude_original_image * np.exp(1j*phase_original_image)
        generalized_image = np.real(np.fft.ifft2(np.fft.fftshift(generalized_freq,axes=(2,1)),axes=(-2, -1)))

        # print(generalized_image.shape)
        # print(type(generalized_image))
        dg_outputs = generalized_image
        # print(generalized_freq.shape)
        dg_fre_outputs = generalized_freq
        (average_PSNR, channel_PSNR) = distance_metric(original_image,dg_outputs)
        PSNR_list.append(average_PSNR)
        channel_PSNR_list.append(channel_PSNR)

    return (PSNR_list,channel_PSNR_list)

# First load training data and test data
# Take one training data and do DG to each domain and get average PSNR

train_dataset, valset = load_dataset(train=True,is_vert_flip = False,is_rotate = False,is_translate = False,is_color_jitter = False,is_DG=0)

original_image,_ = train_dataset.__getitem__(0)
# Unormalize
original_image = original_image / 2 + 0.5
original_image = original_image.cpu().numpy()
# Note the image should be modified into (H,W,C) for plt show.
# plt.imshow(np.transpose(original_image, (1, 2, 0)))
# plt.show()
# for 
print_str = []
sf = 2
lam = 0.5
log_file_name = "./DG_distance_calculate.txt"
# log_file = open(log_file_name,'a')
# log_file.seek(0)
for sf in [0.006,0.01,0.015]:
    for lam in [1,0.75,0.5,0.3]:
        average_PSNR = 0
        for i in range(3):
            test_data_str = ("./data/Pro1-SegmentationData/Domain%d/data/*.bmp") % (i + 1)
            test_label_str = ("./data/Pro1-SegmentationData/Domain%d/label/{}.bmp") % (i + 1)
            if i == 0:
                test_data_str = ("./data/Pro1-SegmentationData/Domain%d/data/*.jpg") % (i + 1)
                test_label_str = ("./data/Pro1-SegmentationData/Domain%d/label/{}.png") % (i + 1)
            testset = load_dataset(train=False,test_data_str = test_data_str, test_label_str = test_label_str)
            (PSNR_list,_) = domain_generization(original_image,testset.x,disp.SSIM_metric,sf,lam)
            average_PSNR_sub = np.mean(PSNR_list)
            average_PSNR = average_PSNR + average_PSNR_sub
            print_str.append(("Scale factor %.3f, lambda %.3f, SSIM for domain %d is: %f \n")%(sf,lam,i+1,np.mean(average_PSNR_sub)))

        print_str.append("%f \n" %( average_PSNR / 3))
        print_str.append("\n")
    # print(testset.x)
for i in print_str:
    with open(log_file_name,'a') as log_file:
        log_file.write(i)

log_file.close()
# print(print_str)

