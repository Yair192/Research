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

x_test_all = np.load('x_test.npy')
y_test_all = np.load('y_test.npy')
print("The X test shape is:", x_test_all.shape)
print("The y test shape is:", y_test_all.shape)

Config = CFG()
RMSE_avg_1_to_IMU_avg = np.zeros([Config.N, Config.IMU_to_train])

device = 'cpu'

for k in range(Config.runs):
    x_test_merged = np.zeros((Config.N, 0, Config.num_of_samples))
    y_test_merged = np.zeros((Config.N, 0))
    input_channels = 0
    for i in range(Config.IMU_to_train):
        print(f'Test run {k + 1} and IMU {i + 1}')
        input_channels += 1
        X, y = x_test_all[i], y_test_all[i]
        x_test_merged = np.concatenate((x_test_merged, X), axis=1)
        y_test_merged = np.concatenate((y_test_merged, y), axis=1)
        x_test_1 = torch.Tensor(x_test_merged)
        y_test_1 = torch.Tensor(y_test_merged)
        model = CNNet.CNN1D(input_channels).to(device)
        state = torch.load(
            f'/home/ystolero/Documents/Research/Simulation/checkpoints_raise_input/run_{k}/1d_cnn_ckpt_{i}.pth',
            map_location=device)
        model.load_state_dict(state['net'])
        model.eval()
        y_pred = model(x_test_1.to(device), input_channels)
        y_pred = y_pred.cpu().detach().numpy()
        RMSE_avg_1_to_IMU_avg[k, i] = rmse_calc(y_pred[:, i], y_test_1[:, 0].numpy())
RMSE_avg_all_running = np.mean(RMSE_avg_1_to_IMU_avg, axis=0)
print(RMSE_avg_all_running * Config.rad_s_to_deg_h)

x_axis = np.arange(1, Config.IMU_to_train + 1)
width = 0.5

# Create a wider figure
plt.figure(figsize=(26, 10))

# Create a bar chart with custom X and Y values
plt.bar(x_axis, RMSE_avg_all_running * Config.rad_s_to_deg_h, width)

# Customize the plot
plt.xticks(x_axis, fontsize=24)
plt.tick_params(labelsize=18)
plt.xlabel('Num of IMU', fontsize=24)
plt.ylabel('RMSE [deg/sec]', fontsize=24)
plt.title('RMSE for simualted data - predicted vs test - as number of trained IMU', fontsize=24)

# Show the plot
plt.show()
