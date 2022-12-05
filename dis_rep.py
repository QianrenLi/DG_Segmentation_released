import numpy as np

def norm_dist(image_1,image_2,ord):
    # Input one image and calculate its absolute eu distance
    return np.linalg.norm(image_1 - image_2,ord)

def CS_dist(image_1,image_2):
    return 0.5 * np.sum(np.devide(np.pow(image_1-image_2,2),image_1+image_2))

# Other option: use the distance metric from sciki-learn
from sklearn.metrics import DistanceMetric
    