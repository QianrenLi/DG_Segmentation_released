# Load Image 
from PIL import Image
from torchvision import transforms
import torch
image_PATH = "./data/Pro1-SegmentationData/Training_data/data/g0004.bmp"
target_PATH = "./data/Pro1-SegmentationData/Training_data/label/g0004.bmp"

data = Image.open(image_PATH,mode='r')
target = Image.open(target_PATH)
seed =240


# Do image transform
input_transform_list = []
label_transform_list = []
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
input_transform_list.append(transforms.ToTensor())
label_transform_list.append(transforms.ToTensor())

# Notice the location of eye are generally in [40: 1500] [0: 400] 
# if is_random_crop:
#     input_transform_list.append(transforms.RandomCrop(600))
#     label_transform_list.append(transforms.RandomCrop(600))

input_transform_list.append(transforms.Resize([32 * 12,32 * 12]))
label_transform_list.append(transforms.Resize([32 * 12, 32 * 12]))
# if i < 5:
input_transform_list.append(transforms.RandomRotation(30,expand=False,fill=0))
label_transform_list.append(transforms.RandomRotation(30,expand=False,fill=1))
# # # if i < 4:
# # #     # 0.1,0.1 is the factor
input_transform_list.append(transforms.RandomAffine(degrees = 0,translate= (0.1,0.1),fill=0))
label_transform_list.append(transforms.RandomAffine(degrees = 0,translate= (0.1,0.1),fill=1))
# # # # if i < 3:
input_transform_list.append(transforms.RandomVerticalFlip())
label_transform_list.append(transforms.RandomVerticalFlip())
input_transform_list.append(transforms.RandomHorizontalFlip())
label_transform_list.append(transforms.RandomHorizontalFlip())
# # # if i < 2:
# input_transform_list.append(transforms.ColorJitter(brightness=0.3,contrast=0.3, saturation= 0.3))
# # # if i < 1:
input_transform_list.append(transforms.Resize([32 * 12, 32 * 12]))
label_transform_list.append(transforms.Resize([32 * 12, 32 * 12]))
# input_transform_list.append(normalize)
# label_transform_list.append(transforms.Grayscale(1))
im_transform = transforms.Compose(
    input_transform_list
)
label_transform = transforms.Compose(
    label_transform_list
)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
transformed_data = im_transform(data)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
transformed_label = label_transform(target)

import matplotlib.pyplot as plt
transformed_data_np = transformed_label.numpy()
# print(transformed_data_np)
plt.imshow(transformed_data_np.transpose((1,2,0)))
plt.show()