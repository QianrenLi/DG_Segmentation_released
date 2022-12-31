
# Train Network with 5 Fold Each
import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self,weight=None,size_average = True):
        super(DiceBCELoss, self).__init__()
    def forward(self,inputs,targets,smooth=1):
        inputs = F.sigmoid(inputs)
        batch_number = targets.size(0)
        # inputs = inputs[:,0,:,:]
        Dice_BCE = 0
        for i in range(1):
            a = inputs[:,i].view(batch_number,-1)
            b = targets[:,i].view(batch_number,-1)
            intersection = (a*b).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (a.sum() + b.sum() + smooth)
            # print(dice_loss)
            Dice_BCE += dice_loss
        # Dice_BCE = Dice_BCE/2
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        # BCE = 0
        Dice_BCE = Dice_BCE + BCE
        return Dice_BCE




from dataset import load_dataset
import time

epoch_number = 20

for train_idx in [0]:
    """
    Train idx = 4: training for domain 1
    Train idx = 5: training for domain 2
    Train idx = 6: training for domain 3
    Train idx = 7: training for dynamic domain
    """
    

    #################### Model Define ######################
    import torch.nn as nn
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    import torch

    div_factor = 5
    loss_list = []


    net = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
    )


    import torch.optim as optim
    import torch.nn as nn
    def reset_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    for i in range(div_factor):
        criterion = DiceBCELoss()
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        net = net.to(device)
        criterion = criterion.to(device)
        net.apply(reset_weights)


        trainset, valset = load_dataset(train=True,is_vert_flip = False,is_rotate = False,is_translate = False,is_color_jitter = False,is_DG=0,div_factor=5,fold_id=i)
        PATH = ('./net_k_fold_%d.pth') % i
        log_file_name = ('./k_fold%d.txt')%(epoch_number)

        
        # trainset, valset = load_dataset(train=True,is_vert_flip = True,is_rotate = True,is_translate = True,is_color_jitter = False,is_DG=False)
        # trainset, valset = load_dataset(train=True,is_DG=False)

        log_file = open(log_file_name,'w')

        ########### Delete file content
        log_file.seek(0)
        log_file.truncate()
        ########### Delete file content



        batch_size = 4

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        # running_loss = 0.0
        for epoch in range(epoch_number):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, (inputs, target) in enumerate(trainloader):
                inputs = inputs.to(device)
                target = target.to(device)
                outputs = net(inputs)
                loss = criterion(outputs,target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # log_file.write(("%f\n") % loss.item())
                # print(("%f\n") % loss.item())
                if i % 20 == 19:    # print every 20 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                    running_loss = 0.0
        

        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
        import numpy as np
        dscs = []
        with torch.no_grad():
            for data in valloader:
                # print(data)
                (images, targets,names) = data
                ############ Activate this for CUDA ##################
                images = images.to(device)
                targets = targets.to(device)
                ############ Activate this for CUDA ##################
                outputs = net(images.to(torch.float32))
                
                inputs = F.sigmoid(outputs)
                batch_number = targets.size(0)
                # inputs = inputs[:,0,:,:]
                smooth = 1
                Dice_BCE = 0
                for j in range(2):
                    a = inputs[:,j].view(batch_number,-1)
                    b = targets[:,j].view(batch_number,-1)
                    intersection = (a*b).sum()
                    dice_loss = 1 - (2. * intersection + smooth) / (a.sum() + b.sum() + smooth)
                    Dice_BCE += dice_loss
                dscs.append(1 - Dice_BCE.cpu().numpy()) 

        dsc_test = np.mean(dscs)
        torch.save(net.state_dict(), PATH)
        print(('Dsc of validation %f')% (dsc_test))
        loss_list.append(dsc_test)
    print(loss_list)
        # log_file.close()
        ################## Temp Model Save ##############
        # torch.save(net.state_dict(), PATH)
        # print('Finished Training')

# Compare Validation loss in different class



# Retrain model