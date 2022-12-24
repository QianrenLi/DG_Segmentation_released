# Calculate the mean and variance
import torch
from torchvision.datasets import ImageFolder
from dataset import load_dataset

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())
def getStatT(Test_data):
    '''
    Compute mean and variance for training data
    :param Test: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(Test_data))
    train_loader = torch.utils.data.DataLoader(
        Test_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(Test_data))
    std.div_(len(Test_data))
    return list(mean.numpy()), list(std.numpy())
    
# To run the program, one must deactivate the normalize

if __name__ == '__main__':
    train_dataset, valset = load_dataset(train=True,is_vert_flip = True,is_rotate = True,is_translate = True,is_color_jitter = False,is_DG=0)

    print(getStat(train_dataset))

    print(getStatT(valset))

    for i in range(3):
        test_data_str = ("./data/Pro1-SegmentationData/Domain%d/data/*.bmp") % (i + 1)
        test_label_str = ("./data/Pro1-SegmentationData/Domain%d/label/{}.bmp") % (i + 1)
        if i == 0:
            test_data_str = ("./data/Pro1-SegmentationData/Domain%d/data/*.jpg") % (i + 1)
            test_label_str = ("./data/Pro1-SegmentationData/Domain%d/label/{}.png") % (i + 1)
        testset = load_dataset(train=False,test_data_str = test_data_str, test_label_str = test_label_str)
        print(getStatT(testset))
    
