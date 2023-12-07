import os
import time
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

seed = 111
np.random.seed(seed)
torch.manual_seed(seed)

from data_generator import CFG
import cnn_1d_model as CNNet

from utils import rmse_calc

x_test_1 = np.load('x_test.npy')
y_test_1 = np.load('y_test.npy')
print("The X test 1st IMU shape is:",x_test_1.shape)
print("The y test 1st IMU shape is:",y_test_1.shape)

Config = CFG()
RMSE_avg_1_to_IMU_avg = np.zeros(Config.IMU_to_train)


device = 'cpu'
model = CNNet.CNN1D(Config.input_channels).to(device)
for i in range(Config.IMU_to_train):
    state = torch.load('/home/ystolero/Documents/Research/Simulation/checkpoints/1d_cnn_ckpt_{i}.pth', map_location=device)
    model.load_state_dict(state['net'])
    model.eval()
    y_pred = model(x_test_1[0].to(device))
    y_pred = y_pred.cpu().detach().numpy()
    RMSE_avg_1_to_IMU_avg[i] = rmse_calc(y_pred, y_test_1[0].numpy())
print(RMSE_avg_1_to_IMU_avg)