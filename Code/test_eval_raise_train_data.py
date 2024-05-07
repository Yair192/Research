import os
import numpy as np
import torch
import cnn_1d_model as CNNet
from matplotlib import pyplot as plt
from data_generator import CFG
from utils import rmse_calc, rmse_model_based_calc, create_segments_and_labels, create_seg_for_single_gyro

seed = 111
np.random.seed(seed)
torch.manual_seed(seed)
Config = CFG()

plt.close('all')
rmse_model_based = rmse_model_based_calc()

x_train_all = np.load('/home/ystolero/Documents/Research/bias_data_from_slam_course_dots/data_array.npy')
y_train_all = np.mean(x_train_all, axis=3)

x_test_1 = x_train_all[0, 90:, :, 0:Config.samples_to_train]
y_test_1 = y_train_all[0, 90:]

# x_test = np.load('/home/ystolero/Documents/Research/Simulation/Code/x_test_real.npy')
# y_test = np.load('/home/ystolero/Documents/Research/Simulation/Code/y_test_real.npy')


# x_test_1_win, y_test_1_win = create_segments_and_labels(np.squeeze(x_test_1, axis=1), np.squeeze(y_test_1, axis=1),
#                                                         Config.window_size, Config.step_size, Config.ratio_window_to_step)

# x_test_1_win, y_test_1_win = create_seg_for_single_gyro(x_test[:Config.samples_to_train], y_test, Config.window_size, Config.step_size)

# x_test_1_win = np.expand_dims(x_test_1_win, axis=1)
# y_test_1_win = np.expand_dims(y_test_1_win, axis=1)
x_test_1_win = torch.Tensor(x_test_1)
y_test_1_win = torch.Tensor(y_test_1)

print("The RMSE model based is:", rmse_model_based)

RMSE_avg_1_to_IMU_avg = np.zeros([Config.runs, Config.IMU_to_train])
device = 'cpu'

for k in range(Config.runs):
    for i in range(Config.IMU_to_train):
        model = CNNet.CNN1D(Config.input_channels).to(device)
        # model = CNNet.ResNet1D().to(device)
        # model = CNNet.LSTMGyro(Config.input_channels, CNNet.lstm_units, CNNet.dense_units, CNNet.output_units)
        state = torch.load(f'/home/ystolero/Documents/Research/Simulation/checkpoints/run_{k}/1d_cnn_ckpt_{i}.pth',
                           map_location=device)
        model.load_state_dict(state['net'])
        model.eval()
        y_pred = model(x_test_1_win.to(device))
        y_pred = y_pred.cpu().detach().numpy()
        RMSE_avg_1_to_IMU_avg[k, i] = rmse_calc(y_pred, y_test_1_win.numpy())

RMSE_avg_all_running = np.mean(RMSE_avg_1_to_IMU_avg, axis=0)
print(RMSE_avg_all_running)

x_axis = np.arange(1, Config.IMU_to_train + 1)
width = 0.5
line_x = np.linspace(0.75, len(x_axis) + 0.25, 40)  # 100 points to span the entire range


# Create a wider figure
plt.figure(figsize=(13, 5))

# Create a bar chart with custom X and Y values
plt.bar(x_axis, RMSE_avg_all_running, width, label=f'NN - {Config.time_for_train} seconds recording')
plt.plot(line_x, [rmse_model_based] * len(line_x), 'r',
         label=f'Model based - {Config.time_for_model_based} seconds recording', linewidth=5)

# Customize the plot
plt.xticks(x_axis, fontsize=24)
plt.tick_params(labelsize=18)
plt.xlabel('Num of gyroscopes', fontsize=24)
plt.ylabel('RMSE [deg/hour]', fontsize=24)
# plt.title('RMSE for simualted data - predicted vs test - as number of trained IMU', fontsize=24)
plt.legend(loc=1, bbox_to_anchor=(1, 1), prop={'size': 24})
# Show the plot
plt.show()

