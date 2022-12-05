import numpy as np

def norm_dist(image_1,image_2,ord):
    # Input one image and calculate its absolute ord-norm distance
    # Here each channel distance will be calculated first, and then do annother norm calculation
    channel_norm = np.zeros((3,1))
    channel_norm[0] = np.linalg.norm(image_1[0,:,:] - image_2[0,:,:],ord = ord)
    channel_norm[1] = np.linalg.norm(image_1[1,:,:] - image_2[1,:,:],ord = ord)
    channel_norm[2] = np.linalg.norm(image_1[2,:,:] - image_2[2,:,:],ord = ord)
    return np.linalg.norm(channel_norm,ord = ord)

def CS_dist(image_1,image_2):
    return 0.5 * np.sum(np.divide(np.power(image_1-image_2,2),image_1+image_2))

# Other option: use the distance metric from sciki-learn
from sklearn.metrics import DistanceMetric
    
def in_cluster_distance(images,distance_metric, **params):
    # Input data size is a 4 channel( N C H W ) numpy array
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

# Test code
images = np.random.rand(5,3,50,50)
# images = np.ones((5,3,50,50))
# in_dis = in_cluster_distance(images,norm_dist,ord = 2)
# in_dis = in_cluster_distance(images,CS_dist)
print(in_dis)