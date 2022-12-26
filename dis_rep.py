import numpy as np

def norm_dist(image_1,image_2,ord):
    # Input two images and calculate its absolute ord-norm distance
    # Here each channel distance will be calculated first, and then do annother norm calculation
    channel_norm = np.zeros((3,1))
    channel_norm[0] = np.linalg.norm(image_1[0,:,:] - image_2[0,:,:],ord = ord)
    channel_norm[1] = np.linalg.norm(image_1[1,:,:] - image_2[1,:,:],ord = ord)
    channel_norm[2] = np.linalg.norm(image_1[2,:,:] - image_2[2,:,:],ord = ord)
    return np.linalg.norm(channel_norm,ord = ord)

def CS_dist(image_1,image_2):
    return 0.5 * np.sum(np.divide(np.power(image_1-image_2,2),image_1+image_2))

def MSE_metric(image_1,image_2):
    """
    @description  : Used to Generate MSE, not directional
    ---------
    @param  :   image_1(3D np array), image_2(3D np array)
    -------
    @Returns  : (MSE_value -> float, channel MSE -> np array)
    -------
    Input is (C H W) metric in range (0~1)
    MSE value is the MSE for all the channel
    The channel MSE is the MSE value of each channel
    The MSE value range: (0~255)
    """


    if np.shape(image_1) != np.shape(image_2):
        print("The shape of two images must be the same")
        return
    (k,m,n) = np.shape(image_1)
    timage_1 = image_1 * 255
    timage_2 = image_2 * 255
    channel_MSE = np.zeros((3,1))
    channel_MSE[0] = np.linalg.norm(timage_1[0,:,:] - timage_2[0,:,:],ord = 2)
    channel_MSE[1] = np.linalg.norm(timage_1[1,:,:] - timage_2[1,:,:],ord = 2)
    channel_MSE[2] = np.linalg.norm(timage_1[2,:,:] - timage_2[2,:,:],ord = 2)

    channel_MSE = np.true_divide(np.power(channel_MSE,2),m*n)
    MSE_value = (channel_MSE[0] + channel_MSE[1] + channel_MSE[2])/3
    return (MSE_value, channel_MSE)
    

def PSNR_metric(image_1,image_2):
    """
    @description  : Used to Generate PSNR, not directional
    ---------
    @param  :   image_1(3D np array), image_2(3D np array)
    -------
    @Returns  : (PSNR_value -> float, channel PSNR -> np array)
    -------
    Range of input image should be (0~1)
    The output PSNR in dB value
    """
    if np.shape(image_1) != np.shape(image_2):
        print("The shape of two images must be the same")
        return
    (MSE_value, channel_MSE) = MSE_metric(image_1,image_2)
    # return (10*np.log10(255/MSE_value),10*np.log10(255/channel_MSE))
    return 10*np.log10(255/MSE_value)

def SSIM_metric(image_1,image_2):
    """
    @description  : Used to Generate SSIM, not directional
    ---------
    @param  : image_1(3D np array), image_2(3D np array)
    -------
    @Returns  : (SSIM value -> float,  channel SSIM -> np array)
    -------
    The input should be (C,H,W) and in RGB format
    Image size must be greater than 7 * 7
    """
    
    
    from skimage.metrics import structural_similarity as ssim
    
    timage_1 = image_1[0,:,:] * 0.3 + image_1[1,:,:] * 0.59 + image_1[2,:,:] * 0.11
    timage_2 = image_2[0,:,:] * 0.3 + image_2[1,:,:] * 0.59 + image_2[2,:,:] * 0.11
    ssim_channel = np.zeros((3,1))
    ssim_channel[0] = ssim(image_1[0,:,:],image_2[0,:,:])
    ssim_channel[1] = ssim(image_1[1,:,:],image_2[1,:,:])
    ssim_channel[2] = ssim(image_1[2,:,:],image_2[2,:,:])

    # return (ssim(timage_1,timage_2), ssim_channel)
    return ssim(timage_1,timage_2)
 
# Other option: use the distance metric from sciki-learn
from sklearn.metrics import DistanceMetric

    
def intra_cluster_distance(images,distance_metric, **params):
    # Input: images is a 4 channel( N C H W ) numpy array
    # Input: distance_metric is the function to calculate distance
    # Input: **params is the required additional parameter for the metric function
    temp_shape = np.shape(images)
    if len(temp_shape) != 4:
        print("Invalid data size")
        return
    elif temp_shape[1] != 3:
        print('Invalid color channel')
        return
    else:
        number = temp_shape[0]
        distance_matrix = np.zeros((number,number))
        for i in range(number):
            for j in range(i+1,number):
                distance_matrix[i,j] = distance_metric(images[i,:,:,:],images[j,:,:,:],**params)
        return 2 * np.sum(distance_matrix)/(number * number)

def inter_cluster_diatance(images_1,images_2, distance_type ,distance_metric, **params):
    # Input: images_1 and images_2 are 4 channel( N C H W ) numpy arrays
    # Input: distance_type contains 3 types: 0 - minimum, 1 - maximum, 2 - mean
    # Input: distance_metric is the function to calculate distance
    # Input: **params is the required additional parameter for the metric function
    # Output: None, if not Valid;
    #           If distance_type == 0 or 1 : minimum distance, and indexs list
    #           If distance_type == 2      : minimum distance, and mean picture 1 and 2
    temp_shape_1 = np.shape(images_1)
    temp_shape_2 = np.shape(images_2)
    if len(temp_shape_1) != 4 or len(temp_shape_2) != 4:
        print("Data should be in [N C H W]")
        return
    elif temp_shape_1[1] != 3 or temp_shape_2[1] != 3:
        print('Invalid color channel')
        return
    elif temp_shape_1[1:] != temp_shape_2[1:]:
        print('Image size of two cluster is different !')
    else:
        if distance_type == 0 or distance_type == 1:
            number_1 = temp_shape_1[0]
            number_2 = temp_shape_2[0]
            distance_matrix = np.zeros((number_1,number_2))
            for i in range(number_1):
                for j in range(number_2):
                    distance_matrix[i,j] = distance_metric(images_1[i,:,:,:],images_2[j,:,:,:],**params)

            if distance_type == 0:
                index = int(distance_matrix.argmin())
                row_num = int(index / number_2)
                col_num = index % number_2
                return distance_matrix[row_num,col_num], [row_num,col_num]

            else:
                index = int(distance_matrix.argmax())
                row_num = int(index / number_2)
                col_num = index % number_2
                return distance_matrix[row_num,col_num], [row_num,col_num]

        elif distance_type == 2:
            mean_image_1 = np.mean(images_1,axis= 0)
            mean_image_2 = np.mean(images_2,axis= 0)
            return distance_metric(mean_image_1,mean_image_2,**params), [mean_image_1, mean_image_2]
        else:
            print("Invalid distance type")
            return None, None

# Test code

# Intra-distance test

# images = np.random.rand(5,3,50,50)
# images = np.ones((5,3,50,50))
# in_dis = intra_cluster_distance(images,norm_dist,ord = 2)
# in_dis = intra_cluster_distance(images,CS_dist)
# print(in_dis)

# Inter-distance test

# images_1 = np.random.rand(5,3,50,50)
# images_2 = np.ones((5,3,50,50))
# in_dis, piciture_index = inter_cluster_diatance(images_1,images_2, 2,norm_dist,ord = 2)
# # in_dis, piciture_index = inter_cluster_diatance(images_1,images_2,CS_dist)
# print(in_dis)
# print(piciture_index)