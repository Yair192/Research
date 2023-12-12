import os
os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
import numpy as np
import pandas as pd

from data_generator import CFG


Config = CFG()


def rmse_calc(pred, real):
    return np.sqrt(np.mean((pred - real) ** 2))


def load_data(imu_to_load, n, num_of_samples, num_of_axis, folder_path):
    x = np.zeros([imu_to_load, n, num_of_axis, num_of_samples])
    y = np.zeros([imu_to_load, n, num_of_axis])
    x_files_paths = [f"{folder_path}IMU_{i}/iteration_{j}_samples.csv" for i in range(1, imu_to_load + 1) for j
                     in range(1, n + 1)]
    y_files_paths = [f"{folder_path}IMU_{i}/IMU_{i}_bias.csv" for i in range(1, imu_to_load + 1)]
    dfs_x = [pd.read_csv(file_path) for file_path in x_files_paths]
    dfs_y = [pd.read_csv(file_path) for file_path in y_files_paths]

    list_counter = 0
    for i in range(imu_to_load):
        for j in range(n):
            x[i, j] = dfs_x[list_counter][:num_of_samples].values
            list_counter += 1

    for i in range(imu_to_load):
        y[i] = dfs_y[i][:n].values

    return x, y


def rmse_model_based_calc():
    # x_test = np.load('/home/ystolero/Documents/Research/Simulation/Code/x_test_real.npy') * Config.deg_s_to_deg_h
    # y_test = np.load('/home/ystolero/Documents/Research/Simulation/Code/y_test_real.npy') * Config.deg_s_to_deg_h
    # x_train_all = np.load('/home/ystolero/Documents/Research/bias_data_from_slam_course_dots/data_array.npy')
    # y_train_all = np.mean(x_train_all, axis=3)
    # x_train_all = np.squeeze(x_train_all, axis=2)
    # y_train_all = np.squeeze(y_train_all, axis=2)

    x_train_all = np.load("x_data_sim.npy") * Config.deg_s_to_rad_s
    y_train_all = np.load("y_data_sim.npy") * Config.deg_s_to_rad_s
    x_test = x_train_all[Config.test_imu_ind, Config.records_to_train:, :Config.samples_to_model_based]
    y_test = y_train_all[Config.test_imu_ind, Config.records_to_train:]
    rmse_model_based = rmse_calc(np.mean(x_test, axis=1), y_test)
    return rmse_model_based


def create_segments_and_labels(X, Y, time_steps, step, ratio):
    x_arr = []
    y_arr = []
    for j in range(0, Config.IMU_to_train):
        segments = np.zeros([int((Config.samples_to_train - time_steps) / step) + 1, Config.window_size])
        labels = np.ones([int((Config.samples_to_train - time_steps) / step) + 1]) * Y[j]
        seg_index = 0
        for i in range(0, Config.samples_to_train - time_steps + 1, step):
            x = X[j, i: i + time_steps]
            segments[seg_index] = x
            seg_index += 1
        x_arr.append(segments)
        y_arr.append(labels)
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    x_arr = np.concatenate(x_arr, axis=0)
    y_arr = np.concatenate(y_arr, axis=0)
    return x_arr, y_arr


def create_seg_for_single_gyro(X, Y, time_steps, step):
    num_of_rec = X.shape[0]
    sample_len = X.shape[1]
    x_arr = []
    y_arr = []
    for j in range(num_of_rec):
        segments = np.zeros([int((sample_len - time_steps) / step) + 1, Config.window_size])
        labels = np.ones([int((sample_len - time_steps) / step) + 1]) * Y[j]
        seg_index = 0
        for i in range(0, sample_len - time_steps + 1, step):
            x = X[j, i: i + time_steps]
            segments[seg_index] = x
            seg_index += 1
        x_arr.append(segments)
        y_arr.append(labels)
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    x_arr = np.concatenate(x_arr, axis=0)
    y_arr = np.concatenate(y_arr, axis=0)
    return x_arr, y_arr
