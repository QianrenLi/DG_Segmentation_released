from cProfile import label
from pyexpat import model
from tkinter.tix import STATUS

from numpy import reshape 
from config import opt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import torch
import models
from data.dataset import Get_data
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer,Netvis
from tqdm import tqdm
import torch.onnx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#plot the figure
import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D # 3D绘图工具包