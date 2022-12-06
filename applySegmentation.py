import torch
# import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import cv2
#load_ext autoreload
#autoreload 2

from dataset import load_dataset


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    testset = load_dataset(train=False)
    batch_size = 4
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)


    PATH = './wxynet-dgDomain2 256_256 numG10 epoch20.pth'
    # torch.save(net.state_dict(), PATH)

    net = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
    )
    net = net.to(device)
    net.load_state_dict(torch.load(PATH))
    dscs = []
    print()
    with torch.no_grad():
        for data in testloader: #length 25
            images, targets, names = data
            images = images.to(device)
            outputs = net(images.to(torch.float32))
            
            for idx, name in enumerate(names):#length 4
                output_np = torch.argmax(outputs[idx], dim=0).cpu().numpy()
                binary_output = np.array(output_np)
                target_np = targets[idx].cpu().numpy().astype(np.uint8)
                
                target_1d = np.reshape(target_np, (-1, 1))
                pred_1d = np.reshape(binary_output, (-1, 1))

                accuracy = accuracy_score(target_1d, pred_1d)
                dsc = f1_score(target_1d, pred_1d)
                
                dscs.append(dsc)
                image = images.cpu().numpy()[-1]
                output = outputs.cpu().numpy()[-1,:,:,:]
                plt.subplot(1,3,1)
                plt.imshow(np.transpose(image,(1,2,0))[:,:,::-1])
                plt.subplot(1,3,2)
                plt.imshow(target_np,cmap='gray')
                plt.subplot(1,3,3)
                plt.imshow(binary_output,cmap='gray')
                # print(name)
                savePath = 'results/wxynet-dgDomain2 256_256 numG10 epoch20/' + name +'.jpg'
                # print(savePath)
                plt.savefig(savePath)
                # plt.show()
    
    dsc_test = np.mean(dscs)
    print('Dsc of test set:', dsc_test)


    # print(len(testloader))
    # print(len(names))
    


