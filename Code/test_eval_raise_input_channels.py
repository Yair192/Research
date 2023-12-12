import os
import numpy as np
import torch
import cnn_1d_model as CNNet
from matplotlib import pyplot as plt
from data_generator import CFG
from utils import rmse_calc, rmse_model_based_calc, create_segments_and_labels

seed = 111
np.random.seed(seed)
torch.cuda.manual_seed(seed)
Config = CFG()

plt.close('all')
os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
x_test_all = np.load('x_test.npy') * Config.deg_s_to_deg_h
y_test_all = np.load('y_test.npy') * Config.deg_s_to_deg_h
rmse_model_based = rmse_model_based_calc()

### Finish check

print("The RMSE model based is:", rmse_model_based)

RMSE_avg_1_to_IMU_avg = np.zeros([Config.runs, Config.IMU_to_train])
device = 'cpu'

for k in range(Config.runs):
    x_test_merged = np.zeros((Config.num_of_windows, 0, Config.window_size))
    y_test_merged = np.zeros((Config.num_of_windows, 0))
    input_channels = 0
    for i in range(Config.IMU_to_train):
        input_channels += 1
        X, y = x_test_all[i, :, :, 0:Config.samples_to_train], y_test_all[i]
        X_win, y_win = create_segments_and_labels(np.squeeze(X, axis=1), np.squeeze(y, axis=1), Config.window_size,
                                                  Config.step_size, Config.ratio_window_to_step)
        X_win = np.expand_dims(X_win, axis=1)
        y_win = np.expand_dims(y_win, axis=1)
        x_test_merged = np.concatenate((x_test_merged, X_win), axis=1)
        y_test_merged = np.concatenate((y_test_merged, y_win), axis=1)
        x_test_win = torch.Tensor(x_test_merged)
        y_test_win = torch.Tensor(y_test_merged)
        model = CNNet.CNN1DRaiseInput(input_channels).to(device)
        # model = CNNet.LSTMGyro(input_channels, CNNet.lstm_units, CNNet.dense_units, input_channels)
        state = torch.load(
            f'/home/ystolero/Documents/Research/Simulation/checkpoints_raise_input/run_{k}/1d_cnn_ckpt_{i}.pth',
            map_location=device)
        model.load_state_dict(state['net'])
        model.eval()
        y_pred = model(x_test_win.to(device))
        y_pred = y_pred.cpu().detach().numpy()
        RMSE_avg_1_to_IMU_avg[k, i] = rmse_calc(y_pred[:, 0], y_test_win.numpy()[:, 0])
RMSE_avg_all_running = np.mean(RMSE_avg_1_to_IMU_avg, axis=0)
print(RMSE_avg_all_running)

x_axis = np.arange(1, Config.IMU_to_train + 1)
width = 0.5

# Create a wider figure
plt.figure(figsize=(13, 5))

# Create a bar chart with custom X and Y values
plt.bar(x_axis, RMSE_avg_all_running, width, label=f'NN - {Config.time_for_train} second recording')
plt.plot(x_axis, [rmse_model_based] * len(x_axis + 2), 'r',
         label=f'Model based - {Config.time_for_model_based} seconds recording')

# Customize the plot
plt.xticks(x_axis, fontsize=24)
plt.tick_params(labelsize=18)
plt.xlabel('Num of IMU', fontsize=24)
plt.ylabel('RMSE [deg/hour]', fontsize=24)
plt.title('RMSE for simualted data - predicted vs test - as number of trained IMU', fontsize=24)
plt.legend(loc=1, prop={'size': 24})
# Show the plot
plt.show()
