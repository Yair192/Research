import os
import numpy as np
import torch
import cnn_1d_model as CNNet
import pickle
from matplotlib import pyplot as plt
from data_generator import CFG
from utils import rmse_calc, rmse_model_based_calc, create_seg_for_single_gyro, running_avg



seed = 111
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

Config = CFG()



def find_indices(arr, value):
    indices = []
    for i in range(len(arr)):
        if np.round(arr[i], 4) == np.round(value, 4):
            indices.append(i)
    return indices

plt.close('all')
rmse_model_based = rmse_model_based_calc(Config.samples_to_model_based)
rmse_end_of_record = rmse_model_based_calc(Config.num_of_samples)


x_train_all = np.load("/home/ystolero/Documents/Research/data/x_data_sim.npy") * Config.deg_s_to_rad_s
y_train_all = np.load("/home/ystolero/Documents/Research/data/y_data_sim.npy") * Config.deg_s_to_rad_s

x_test_all_samples = x_train_all[:, Config.records_to_train:]
x_test_1 = x_train_all[Config.test_imu_ind, Config.records_to_train:, :, 0:Config.samples_to_train]
y_test_1 = y_train_all[Config.test_imu_ind, Config.records_to_train:]

x_test_2 = x_train_all[1, Config.records_to_train:, :, 0:Config.samples_to_train]
y_test_2 = y_train_all[1, Config.records_to_train:]

x_test_3 = x_train_all[2, Config.records_to_train:, :, 0:Config.samples_to_train]
y_test_3 = y_train_all[2, Config.records_to_train:]

x_test_4 = x_train_all[3, Config.records_to_train:, :, 0:Config.samples_to_train]
y_test_4 = y_train_all[3, Config.records_to_train:]

y_mean_all = np.mean(y_test_1, axis=0)
y_mean_all_arr = np.empty([Config.input_channels, 13000])
y_mean_all_arr[0] = y_mean_all[0]
y_mean_all_arr[1] = y_mean_all[1]
y_mean_all_arr[2] = y_mean_all[2]

x_test_1_win, y_test_1_win = create_seg_for_single_gyro(x_test_1, y_test_1, Config.window_size, Config.step_for_train)
x_test_2_win, y_test_2_win = create_seg_for_single_gyro(x_test_2, y_test_2, Config.window_size, Config.step_for_train)
x_test_3_win, y_test_3_win = create_seg_for_single_gyro(x_test_3, y_test_3, Config.window_size, Config.step_for_train)
x_test_4_win, y_test_4_win = create_seg_for_single_gyro(x_test_4, y_test_4, Config.window_size, Config.step_for_train)

x_test_1 = torch.Tensor(x_test_1_win)
y_test_1 = torch.Tensor(y_test_1_win)

x_test_2 = torch.Tensor(x_test_2_win)
y_test_2 = torch.Tensor(y_test_2_win)

x_test_3 = torch.Tensor(x_test_3_win)
y_test_3 = torch.Tensor(y_test_3_win)

x_test_4 = torch.Tensor(x_test_4_win)
y_test_4 = torch.Tensor(y_test_4_win)

print("The RMSE model based is:", rmse_model_based / Config.deg_s_to_rad_s)
print(f"The RMSE at the end of {Config.t} sec is:", rmse_end_of_record / Config.deg_s_to_rad_s)

RMSE_avg_1_to_IMU_avg = np.zeros([Config.runs, Config.IMU_to_train])
NN_vs_model_percentage = np.zeros(Config.IMU_to_train)
y_pred_mean = []
device = 'cpu'

for k in range(Config.runs):
    j = 0
    in_ch = 3
    for i in range(Config.IMU_to_train):

        model = CNNet.MultiHeadCNN(in_ch).to(device)
        state = torch.load(f'/home/ystolero/Documents/Research/checkpoints/run_{k}/1d_cnn_ckpt_{i}.pth',
                           map_location=device)
        model.load_state_dict(state['net'])

        model.eval()
        if i == 0:
            y_pred = model(x_test_1.to(device))
        elif i == 1:
            y_pred = model(torch.cat((x_test_1, x_test_2), dim=1).to(device))
        elif i == 2:
            y_pred = model(torch.cat((x_test_1, x_test_2, x_test_3), dim=1).to(device))
        elif i == 3:
            y_pred = model(torch.cat((x_test_1, x_test_2, x_test_3, x_test_4), dim=1).to(device))

        y_pred = y_pred.cpu().detach().numpy()
        y_pred_mean.append(np.mean(y_pred, axis=0))

        j += 1
        in_ch += 3

RMSE_avg_all_running = np.mean(RMSE_avg_1_to_IMU_avg, axis=0)

print(f"The test loss is {RMSE_avg_all_running**2}")
print(f"The NN RMSE is{RMSE_avg_all_running / Config.deg_s_to_rad_s}")


print(f"NN vs model based percetnage is {NN_vs_model_percentage}")


# x_axis = np.arange(1, Config.IMU_to_train + 1)
# width = 0.5
# line_x = np.linspace(0.75, len(x_axis) + 0.25, 40)  # 100 points to span the entire range
#
#
# # Create a wider figure
# plt.figure(figsize=(13, 5))
#
# # Create a bar chart with custom X and Y values
# plt.bar(x_axis, RMSE_avg_all_running  / Config.deg_s_to_rad_s, width, label=f'NN - {Config.time_for_train} seconds recording')
# plt.plot(line_x, [rmse_model_based / Config.deg_s_to_rad_s] * len(line_x), 'r',
#          label=f'Model based - {Config.time_for_model_based} seconds recording', linewidth=1)
# # Customize the plot
# plt.xticks(x_axis, fontsize=24)
# plt.tick_params(labelsize=18)
# plt.xlabel('Num of gyroscopes', fontsize=24)
# plt.ylabel('RMSE [deg/sec]', fontsize=24)
# plt.legend(loc=1, bbox_to_anchor=(1, 1), prop={'size': 24})
# # Show the plot
# plt.show()

# Print the convergence graph with the prediction points for each IMU

#
x_test_1_all_samples_avg = np.mean(x_test_all_samples[Config.test_imu_ind], axis=0)
x_test_running_avg = running_avg(x_test_1_all_samples_avg)
L1_RA_X_test = rmse_calc(x_test_running_avg, y_mean_all_arr) / Config.deg_s_to_rad_s


plt.plot(Config.t_arr, L1_RA_X_test, label=r"$\mu[t]$")
plt.scatter(Config.t_check, rmse_calc(y_pred_mean[0], y_mean_all) / Config.deg_s_to_rad_s, label = "1 IMU")
plt.scatter(Config.t_check, rmse_calc(y_pred_mean[1][:3] ,y_mean_all) / Config.deg_s_to_rad_s, label = "2 IMUs")
plt.scatter(Config.t_check, rmse_calc(y_pred_mean[2][:3] , y_mean_all) / Config.deg_s_to_rad_s, label = "3 IMUs")
plt.scatter(Config.t_check, rmse_calc(y_pred_mean[3][:3] , y_mean_all) / Config.deg_s_to_rad_s, label = "4 IMUs")


plt.grid(True)
plt.legend()
plt.xlabel('Calibration Time [sec]')
plt.ylabel('RMSE [deg/sec]')
#
# # Define the folder path
# folder_path = '/home/ystolero/Documents/Research/Graphs/spark_fun_data/'
#
# # Save the figure in the specified folder as a .fig file
# plt.savefig(folder_path + 'Raise_imu_for_3_channels_each_IMU.jpeg')

plt.show()

IMU_1_vs_IMU_4 = ((rmse_calc(y_pred_mean[0], y_mean_all) - rmse_calc(y_pred_mean[3][:3], y_mean_all)) / rmse_calc(y_pred_mean[0], y_mean_all)) * 100
print(f"The improvement of 4 IMU against 1 IMU is {IMU_1_vs_IMU_4}")




for i in range(Config.IMU_to_train):
    index = find_indices(L1_RA_X_test, rmse_calc(y_pred_mean[i][:3], y_mean_all))[-1] / Config.f

    percentage_of_improve = ((index - Config.t_check) / index) * 100

    print(f"The percentage of improvement for {i+1} IMU is {percentage_of_improve}")
