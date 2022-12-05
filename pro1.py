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
#load_ext autoreload
#autoreload 2

from dataset import load_dataset

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    trainset, valset = load_dataset(train=True)
    testset = load_dataset(train=False)

    batch_size = 4

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    print(valloader)




    # def imshow(img):
    #     # The data need to be normalized and unnormalized to keep the same
    #     # img = img / 255
    #     # img = img / 2 + 0.5     # unnormalize

    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    # # get some random training images
    # dataiter = iter(valloader)
    # images, targets, _ = next(dataiter)
    # print(targets)

    # temp_image = torchvision.utils.make_grid(images)


    # # show images
    # imshow(torchvision.utils.make_grid(targets))




    net = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
    )
    # print(net)
    net = net.to(device)


    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, target) in enumerate(trainloader):
            
            # zero the parameter gradients
            optimizer.zero_grad()

            inputs = inputs.to(device)
            target = target.to(device)

            # forward + backward + optimizeo 
            # change the data to float type
            outputs = net(inputs.to(torch.float32))
            # print((outputs.shape))
            # print(target.to(torch.float32).dtype)
            loss = criterion(outputs, target.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './wxynet.pth'
    torch.save(net.state_dict(), PATH)

    net = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
    )
    net = net.to(device)
    net.load_state_dict(torch.load(PATH))
    dscs = []

    with torch.no_grad():
        for data in testloader:
            images, targets, names = data
            images = images.to(device)
            outputs = net(images.to(torch.float32))
            
            for idx, name in enumerate(names):
                output_np = torch.argmax(outputs[idx], dim=0).cuda().numpy()
                binary_output = np.array(output_np)
                target_np = targets[idx].cuda().numpy().astype(np.uint8)
                
                target_1d = np.reshape(target_np, (-1, 1))
                pred_1d = np.reshape(binary_output, (-1, 1))

                accuracy = accuracy_score(target_1d, pred_1d)
                dsc = f1_score(target_1d, pred_1d)
                
                dscs.append(dsc)

    dsc_test = np.mean(dscs)
    print('Dsc of test set:', dsc_test)