import numpy as np
import pandas as pd


def rmse_calc(pred, real):
    return np.sqrt(np.mean(np.sum((pred - real) ** 2, axis=0)))


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