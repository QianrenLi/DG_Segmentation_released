# Validation, training DICE, HD95
Cross_entropy_accuracy = [0.171867,0.080002,0.183476,0.147741,22.197613,4.070576,7.992366]
Cross_entropy_accuracy_by_TA = [0.996240,0.997694,0.996052]
DICE_accuracy = [0.872547,0.653809,0.868984, 0.780455, 3.846430,2.402280,4.130500]
DICE_data_argmentation_accuracy = [0.067608,0.033740,0.071058,0.068719,3.353747,13.901554,34.713509]

DICE_cross_entropy = [ 0.892553,0.654374,0.898912,0.804685,13.055154,1.948374,4.139773]

DICE_cross_entropy_agu =  [0.910399,0.732954,0.912890,0.865212,4.166193,1.534257,2.486036]

DICE_one_channel = [0.403746,0.114510,0.362513,0.288859,11.689443,2.596307,4.136328]

DG_accuracy = [0.901450,0.714748,0.907067,0.849088,3.707399,1.618404,2.820824]

DG_agu_accuracy = [0.907357,0.736940,0.908792,0.889355,3.722626,1.625574,1.891746]

import matplotlib.pyplot as plt

#Group Bar chart

import random
import numpy as np
import matplotlib.pyplot as plt
#生成信息
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'r', 'b']
labels = ['Vlidation','Test1','Test2','Test3','Test1','Test2','Test3']
labels1 = labels[0:4]


############## 2 Plot #####################
data1 = DICE_cross_entropy[0:4]
data2 = DICE_cross_entropy_agu[0:4]
############## 2 Plot #####################

############## 2 Plot #####################
data1 = DICE_cross_entropy_agu[0:4]
data2 = DG_agu_accuracy[0:4]
############## 2 Plot #####################

############## 2 Plot #####################
data1 = DICE_cross_entropy[0:4]
data2 = DG_accuracy[0:4]
############## 2 Plot #####################

############## 2 Plot #####################
data2 = DICE_accuracy[0:4]
data1 = DICE_one_channel[0:4]
############## 2 Plot #####################

# ############ 3 plot #####################
# data1 = Cross_entropy_accuracy[0:4]
# data2 = DICE_accuracy[0:4]
# data3 = DICE_cross_entropy[0:4]
# ############ 3 plot #####################

offset = 1
width = 1.5
pos_diff = 5
xpos = np.arange(0,pos_diff* 4,pos_diff)


fig, ax = plt.subplots(figsize=(10,8))

############## 2 Plot #####################
bars1 = plt.bar(xpos-width/2, data1, align='center', width=width, alpha=0.9, color='#1f77b4', edgecolor='black',label = 'DICE + BCE/DICE Loss')
bars2 = plt.bar(xpos+width/2, data2, align='center', width=width, alpha=0.9, color='#ff7f0e', edgecolor='black',label = 'DICE + BCE + Data Augment/DICE Loss')
############## 2 Plot #####################

############## 2 Plot #####################
bars1 = plt.bar(xpos-width/2, data1, align='center', width=width, alpha=0.9, color='#1f77b4', edgecolor='black',label = 'Without DG/DICE Loss')
bars2 = plt.bar(xpos+width/2, data2, align='center', width=width, alpha=0.9, color='#ff7f0e', edgecolor='black',label = 'DG/DICE Loss')
############## 2 Plot #####################

############## 2 Plot #####################
bars1 = plt.bar(xpos-width/2, data1, align='center', width=width, alpha=0.9, color='#1f77b4', edgecolor='black',label = 'DICE 1 channel/DICE Loss')
bars2 = plt.bar(xpos+width/2, data2, align='center', width=width, alpha=0.9, color='#ff7f0e', edgecolor='black',label = 'DICE 2 channel/DICE Loss')
############## 2 Plot #####################

# ############# 3 plot #####################
# bars1 = plt.bar(xpos-width , data1, align='center', width=width, alpha=0.9, color='#1f77b4',edgecolor='black', label = 'Cross Entropy/DICE loss')
# bars2 = plt.bar(xpos, data2, align='center', width=width, alpha=0.9, color='#ff7f0e',edgecolor='black', label = 'DICE/DICE loss')
# bars3 = plt.bar(xpos+ width , data3, align='center', width=width, alpha=0.9, color='#2ca02c',edgecolor='black', label = 'DICE + Cross Entropy/DICE loss')
# ############# 3 plot #####################

ax.set_xticks(xpos) #确定每个记号的位置
ax.set_xticklabels(labels1)  #确定每个记号的内容
ax.set_ylabel("DICE accuracy",fontsize='x-large',fontweight='heavy')
ax2 = ax.twinx()
x2pos = np.arange(pos_diff * 4 + offset,pos_diff * 7 + offset,pos_diff)
labels2 = labels[4:]

############## 2 Plot #####################
data1 = DICE_cross_entropy[4:]
data2 = DICE_cross_entropy_agu[4:]
############## 2 Plot #####################

############## 2 Plot #####################
data1 = DICE_cross_entropy_agu[4:]
data2 = DG_agu_accuracy[4:]
############## 2 Plot #####################

############## 2 Plot #####################
data1 = DICE_cross_entropy[4:]
data2 = DG_accuracy[4:]
############## 2 Plot #####################

############## 2 Plot #####################
data2 = DICE_accuracy[4:]
data1 = DICE_one_channel[4:]
############## 2 Plot #####################

# ############## 3 plot #####################
# data1 = Cross_entropy_accuracy[4:]
# data2 = DICE_accuracy[4:]
# data3 = DICE_cross_entropy[4:]
# ############## 3 plot #####################


############## 2 Plot #####################
bars3 = plt.bar(x2pos-width/2, data1, align='center', width=width, alpha=0.9, color='#1f77b4',hatch = '/',edgecolor='black', label = 'DICE + BCE/HD95')
bars4 = plt.bar(x2pos+width/2, data2, align='center', width=width, alpha=0.9, color='#ff7f0e',hatch = '/',edgecolor='black', label = 'DICE + BCE + Data Augment/HD95')
############## 2 Plot #####################

############## 2 Plot #####################
bars3 = plt.bar(x2pos-width/2, data1, align='center', width=width, alpha=0.9, color='#1f77b4',hatch = '/',edgecolor='black', label = 'Without DG/HD95')
bars4 = plt.bar(x2pos+width/2, data2, align='center', width=width, alpha=0.9, color='#ff7f0e',hatch = '/',edgecolor='black', label = 'DG/HD95')
############## 2 Plot #####################

############## 2 Plot #####################
bars3 = plt.bar(x2pos-width/2, data1, align='center', width=width, alpha=0.9, color='#1f77b4',hatch = '/',edgecolor='black', label = 'DICE 1 channel/HD95')
bars4 = plt.bar(x2pos+width/2, data2, align='center', width=width, alpha=0.9, color='#ff7f0e',hatch = '/',edgecolor='black', label = 'DICE 2 channel/HD95')
############## 2 Plot #####################

# ############## 3 plot #####################
# bars4 = plt.bar(x2pos-width , data1, align='center', width=width, alpha=0.9, color='#1f77b4',hatch = '/',edgecolor='black',label = 'Cross Entropy/HD95')
# bars5 = plt.bar(x2pos, data2, align='center', width=width, alpha=0.9, color='#ff7f0e', hatch = '/',edgecolor='black',label = 'DICE/HD95')
# bars6 = plt.bar(x2pos+ width , data3, align='center', width=width, alpha=0.9, color='#2ca02c',hatch = '/',edgecolor='black', label = 'DICE + Cross Entropy/HD95')
# ############## 3 plot #####################
ax2.set_xticks(np.concatenate((xpos,x2pos),axis=None))
ax2.set_xticklabels(labels) 
ax2.set_ylabel("Hausdorff Distance 95%",fontsize='x-large',fontweight='heavy')
# plt.legend()
#给每个柱子上面添加标注
def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
              xy=(rect.get_x() + rect.get_width() / 2, height),
              xytext=(0, 3),  # 3 points vertical offset
              textcoords="offset points",
              ha='center', va='bottom'
              )

############## 2 plot #####################             
autolabel(bars1,ax)
autolabel(bars2,ax)
autolabel(bars3,ax2)
autolabel(bars4,ax2)
############## 2 plot #####################

# ############## 3 plot #####################             
# autolabel(bars1,ax)
# autolabel(bars2,ax)
# autolabel(bars3,ax)
# autolabel(bars4,ax2)
# autolabel(bars5,ax2)
# autolabel(bars6,ax2)
# ############## 3 plot #####################

#展示结果

# Set y axis range
# ax.set_ylim((0.5,1))
lns = [bars1,bars2,bars3,bars4]
# lns = [bars1,bars2,bars3,bars4,bars5,bars6]
labels = [l.get_label() for l in lns]
plt.legend(lns,labels)


# plt.title("Accuracy of \"Cross entropy loss\", \"DICE loss\" and \"DICE + Cross entropy loss\"",fontsize='xx-large',fontweight='heavy')
# plt.title("Accuracy of \"Domain Generalization\"and \"Without Domain Generalization\"",fontsize='xx-large',fontweight='heavy')
plt.title("Accuracy of \"DICE 2 channel\"and \"DICE 1 channel\"",fontsize='xx-large',fontweight='heavy')
plt.show()

