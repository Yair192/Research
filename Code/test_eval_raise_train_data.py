import os
import numpy as np
import torch
import cnn_1d_model as CNNet
from matplotlib import pyplot as plt
from data_generator import CFG
from utils import rmse_calc, rmse_model_based_calc, create_seg_for_single_gyro, running_avg, rmse_for_every_sample, running_avg_rmse



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

x_test_merged = np.zeros((0, Config.window_size))
y_test_merged = np.zeros(0)

x_train_all = np.load("x_data_sim.npy") * Config.deg_s_to_rad_s
y_train_all = np.load("y_data_sim.npy") * Config.deg_s_to_rad_s


x_test_1_all_samples = x_train_all[Config.test_imu_ind, Config.records_to_train:]
x_test_1 = x_train_all[Config.test_imu_ind, Config.records_to_train:, 0:Config.samples_to_train]
y_test_1 = y_train_all[Config.test_imu_ind, Config.records_to_train:]
y_mean_all = np.mean(y_test_1)
x_test_1_win, y_test_1_win = create_seg_for_single_gyro(x_test_1, y_test_1, Config.window_size, Config.step_for_train)





x_test_1 = torch.Tensor(x_test_1_win)
y_test_1 = torch.Tensor(y_test_1_win)




print("The RMSE model based is:", rmse_model_based / Config.deg_s_to_rad_s)
print(f"The RMSE at the end of {Config.t} sec is:", rmse_end_of_record / Config.deg_s_to_rad_s)

RMSE_avg_1_to_IMU_avg = np.zeros([Config.runs, Config.IMU_to_train])
NN_vs_model_percentage = np.zeros(Config.IMU_to_train)
y_pred_mean = np.zeros(Config.IMU_to_train)
device = 'cpu'

for k in range(Config.runs):
    j = 0
    for i in range(Config.IMU_to_train):

    # for i in Config.imu_to_train:
        model = CNNet.CNN1D(Config.input_channels).to(device)
        state = torch.load(f'/home/ystolero/Documents/Research/Simulation/checkpoints/run_{k}/1d_cnn_ckpt_{i}.pth',
                           map_location=device)
        model.load_state_dict(state['net'])

        model.eval()

        y_pred = model(x_test_1.to(device))
        y_pred = y_pred.cpu().detach().numpy()
        y_pred_mean[i] = np.mean(y_pred)
        RMSE_avg_1_to_IMU_avg[k, j] = rmse_calc(y_pred, y_test_1.numpy())

        j += 1

RMSE_avg_all_running = np.mean(RMSE_avg_1_to_IMU_avg, axis=0)

print(f"The test loss is {RMSE_avg_all_running**2}")
print(f"The NN RMSE is{RMSE_avg_all_running / Config.deg_s_to_rad_s}")

NN_vs_model_percentage = (RMSE_avg_all_running - rmse_model_based) / rmse_model_based *100

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
x_test_1_all_samples_avg = np.mean(x_test_1_all_samples, axis=0)
x_test_running_avg = running_avg(x_test_1_all_samples_avg)
L1_RA_X_test = np.abs(x_test_running_avg - y_mean_all) / Config.deg_s_to_rad_s


plt.plot(Config.t_arr, L1_RA_X_test)
plt.scatter(Config.t_check, np.abs(y_pred_mean[0] - y_mean_all) / Config.deg_s_to_rad_s, label = "1 IMU")
# plt.scatter(Config.t_check, np.abs(y_pred_mean[1] - y_mean_all) / Config.deg_s_to_rad_s, label = "L1 - 2")
plt.scatter(Config.t_check, np.abs(y_pred_mean[2] - y_mean_all) / Config.deg_s_to_rad_s, label = "2 IMU")
# plt.scatter(Config.t_check, np.abs(y_pred_mean[3] - y_mean_all) / Config.deg_s_to_rad_s, label = "L1 - 4")
# plt.scatter(Config.t_check, np.abs(y_pred_mean[4] - y_mean_all) / Config.deg_s_to_rad_s, label = "L1 - 5")
plt.scatter(Config.t_check, np.abs(y_pred_mean[5] - y_mean_all) / Config.deg_s_to_rad_s, label = "3 IMU")

# plt.scatter(5, 0.00124874, label = "L1 - 5 sec")
# plt.scatter(10, 0.00120255, label = "L1 - 10 sec - Data from 1 Axis only")
# plt.scatter(10, 4.35437566e-05, label = "L1 - 10 sec - When averaging 12 axis into 1 VIMU")
# # plt.scatter(20, 0.00053599, label = "L1 - 20 sec")
# # plt.scatter(30, 0.00109949, label = "L1 - 30 sec")
# plt.scatter(40, 0.00167128, label = "L1 - 40 sec")
# plt.scatter(50, 0.00073769, label = "L1 - 50 sec")
# plt.scatter(60, 0.00150552, label = "L1 - 60 sec")


plt.grid(True)
plt.legend()
plt.xlabel('Time [sec]')
# plt.ylabel('L1 [deg/sec]')
plt.show()

for i in range(Config.IMU_to_train):
    index = find_indices(L1_RA_X_test, np.abs(y_pred_mean - y_mean_all)[i])[-1] / Config.f

    percentage_of_improve = ((index - Config.t_check) / index) * 100

    print(f"The percentage of improvement for {i+1} IMU is {percentage_of_improve}")
