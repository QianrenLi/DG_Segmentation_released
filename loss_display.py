import matplotlib.pyplot as plt
import numpy as np

def loss_read(log_file_name):
    f = open(log_file_name,mode='r')
    line = f.readline() # 调用文件的 readline()方法 
    i = 0
    loss = np.zeros((400,1))
    while line:
        loss[i] = float(line)
        line = f.readline()
        i = i +1
    f.close()
    return loss

epoch_number = 20

# ################### Log File Read ########################
# log_file_name = ('./results/Loss/no_DG_epoch_DICE_%d.txt')%(epoch_number)
# log_file_name = ('./results/Loss/DG_epoch_%d.txt')%(epoch_number)
# log_file_name = ('./results/Loss/Data_arg_no_DG_DICE_epoch_%d.txt')%(epoch_number)
log_file_name = ('./results/Loss/Data_arg_no_DG_epoch_%d.txt') % (epoch_number)
# log_file_name = ('./results/Loss/Data_DG_D1_epoch_%d.txt')%(epoch_number)
# log_file_name = ('./results/Loss/Data_arg_DG_epoch_s03_r1_t%d.txt')%(epoch_number)
# log_file_name = ('./results/Loss/Data_arg_DG_D1_epoch_%d.txt')%(epoch_number)
# log_file_name = ('./results/Loss/Data_arg_DG_D2_epoch_%d.txt')%(epoch_number)
# log_file_name = ('./results/Loss/Data_arg_DG_D3_epoch_%d.txt')%(epoch_number)
# log_file_name = ('./results/Loss/Data_DG_D1_epoch_%d.txt')%(epoch_number)
# ################### Log File Read ########################

loss_DICE = loss_read(log_file_name)
no_DG_epoch = ('./results/Loss/no_DG_epoch_DICEBCE_%d.txt')%(epoch_number)
loss_DICE_CE = loss_read(no_DG_epoch)

no_DG_cross_entropy = ('./results/Loss/no_DG_epoch_cross_entropy_%d.txt')%(epoch_number)
loss_CE = loss_read(no_DG_cross_entropy)

x_train_loss = np.array(range(400))
plt.figure()
plt.xlabel('Epoch',fontsize='large',fontweight='heavy')
plt.ylabel('Loss',fontsize='large',fontweight='heavy')

# #################### Compare Data Augment #####################
# plt.plot(x_train_loss/20, loss_DICE, linewidth=1, linestyle="solid", label="Data Augment")
# plt.plot(x_train_loss/20, loss_DICE_CE, linewidth=1, linestyle="solid", label="Without Data Augment")
# #################### Compare Data Augment #####################

# PATH = ('./results/Loss/no_DG_epoch_DICEBCE_%d.txt')%(epoch_number)
# loss1 = loss_read(PATH)
# PATH = ('./results/Loss/DG_epoch_%d.txt') % (epoch_number)
# loss2 = loss_read(PATH)
# PATH = ('./results/Loss/Data_arg_no_DG_epoch_%d.txt') % (epoch_number)
# loss1 = loss_read(PATH)
# PATH = ('./results/Loss/Data_arg_DG_epoch_s03_r05_t%d.txt') % (epoch_number)
# loss2 = loss_read(PATH)


# #################### Compare Domain Generalization #####################
# plt.plot(x_train_loss/20, loss1, linewidth=1, linestyle="solid", label="Without DG")
# plt.plot(x_train_loss/20, loss2, linewidth=1, linestyle="solid", label="DG")
# #################### Compare Domain Generalization #####################

PATH = ('./results/Loss/no_DG_epoch_DICE_%d.txt')%(epoch_number)
loss2 = loss_read(PATH)
PATH = ('./results/Loss/no_DG_epoch_DICE_t%d.txt') % (epoch_number)
loss1 = loss_read(PATH)

#################### Compare DICE implementation #####################
plt.plot(x_train_loss/20, loss1, linewidth=1, linestyle="solid", label="DICE 1 channel")
plt.plot(x_train_loss/20, loss2, linewidth=1, linestyle="solid", label="DICE 2 channel")
#################### Compare DICE implementation #####################

#################### Compare Loss Function ####################
# plt.plot(x_train_loss/20, loss_DICE/2, linewidth=1, linestyle="solid", label="DICE Loss")
# plt.plot(x_train_loss/20, loss_DICE_CE/3, linewidth=1, linestyle="solid", label="DICE Loss + Cross Entropy Loss")
# plt.plot(x_train_loss/20, loss_CE, linewidth=1, linestyle="solid", label="Cross Entropy Loss")
#################### Compare Loss Function ####################

Arg_DICE = ('./results/Loss/Data_arg_no_DG_DICE_epoch_%d.txt')%(epoch_number)
loss_DICE_arg = loss_read(Arg_DICE)
# plt.plot(x_train_loss/20, loss_DICE_arg/2, linewidth=1, linestyle="solid", label="DICE + Data Augment")
ax = plt.gca()  # 获取当前图像的坐标轴信息
ax.set_xticks(range(0, 21,5)) 
plt.legend()
plt.grid(linestyle='-.')
plt.title('Loss Curve',fontsize='xx-large',fontweight='heavy')
plt.show()







 
