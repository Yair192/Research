import os
import numpy as np
import torch
import cnn_1d_model as CNNet
from matplotlib import pyplot as plt
from data_generator import CFG
from utils import rmse_calc, rmse_model_based_calc

seed = 111
np.random.seed(seed)
torch.manual_seed(seed)

Config = CFG()

os.chdir('/tmp/pycharm_project_389/Simulation/Code')
x_test_all = np.load('x_test.npy')
y_test_all = np.load('y_test.npy')
rmse_model_based = rmse_model_based_calc()



x_test_1 = torch.Tensor(x_test_all[0, :, :, 0:Config.samples_to_train])
y_test_1 = torch.Tensor(y_test_all[0])

print("The RMSE model based is:", rmse_model_based * Config.rad_s_to_deg_h)

RMSE_avg_1_to_IMU_avg = np.zeros([Config.N, Config.IMU_to_train])
device = 'cpu'
for k in range(Config.runs):
    for i in range(Config.IMU_to_train):
        model = CNNet.CNN1D(Config.input_channels).to(device)
        state = torch.load(
            f'/home/ystolero/Documents/Research/Simulation/checkpoints_avg_on_data/run_{k}/1d_cnn_ckpt_{i}.pth',
            map_location=device)
        model.load_state_dict(state['net'])
        model.eval()
        y_pred = model(x_test_1.to(device))
        y_pred = y_pred.cpu().detach().numpy()
        RMSE_avg_1_to_IMU_avg[k, i] = rmse_calc(y_pred, y_test_1.numpy())

RMSE_avg_all_running = np.mean(RMSE_avg_1_to_IMU_avg, axis=0)
print(RMSE_avg_all_running * Config.rad_s_to_deg_h)

x_axis = np.arange(1, Config.IMU_to_train + 1)
width = 0.5

# Create a wider figure
plt.figure(figsize=(26, 10))

# Create a bar chart with custom X and Y values
plt.bar(x_axis, RMSE_avg_all_running * Config.rad_s_to_deg_h, width, label='NN - 30 second recording')
plt.plot(x_axis, [rmse_model_based * Config.rad_s_to_deg_h] * len(x_axis + 2), 'r',
         label='Model based - 5 Minutes recording')

# Customize the plot
plt.xticks(x_axis, fontsize=24)
plt.tick_params(labelsize=18)
plt.xlabel('Num of IMU', fontsize=24)
plt.ylabel('RMSE [deg/sec]', fontsize=24)
plt.title('RMSE for simualted data - predicted vs test - as number of trained IMU', fontsize=24)
plt.legend(loc=1, prop={'size': 24})
# Show the plot
plt.show()
