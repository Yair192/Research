import os
import numpy as np
import pandas as pd
import torch
os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')


class CFG:
    def __init__(self):
        self.f = 120
        self.t = 108
        self.time_for_train = 30
        self.time_for_model_based = 60
        self.time_for_train_window = 30
        self.time_for_test_window = 60
        self.time_for_train_step = 30
        self.time_for_test_step = 30
        self.N_to_simulate = 100
        self.N_to_load = 100
        self.num_of_samples = self.f * self.t
        self.samples_to_model_based = self.f * self.time_for_model_based
        self.samples_to_train = self.f * self.time_for_train
        self.window_size = self.time_for_train_window * self.f
        self.step_size = self.time_for_test_step * self.f
        self.step_for_train = self.time_for_train_step * self.f
        self.ratio_window_to_step = int(self.window_size / self.step_size)
        self.ratio_window_to_step_for_train = int(self.window_size / self.step_for_train)
        self.num_of_windows_train = (int((self.samples_to_train - self.window_size) / self.step_for_train) + 1) * self.N_to_load
        self.num_of_windows_test = (int((self.samples_to_train - self.window_size) / self.step_size) + 1) * self.N_to_load
        self.IMU_to_simulate = 14
        self.IMU_to_test = 1

        self.IMU_to_train = 4
        self.imu_to_train = [0, 1, 2, 3, 4, 5, 6, 7]
        self.test_imu_ind = 0

        self.runs = 1
        self.deg_h_to_deg_s = 1 / 3600
        self.deg_s_to_deg_h = 1 / self.deg_h_to_deg_s
        self.bias = 0.02666096
        self.bias_std = 0
        self.noise_std = 0.1
        self.num_of_axis = 1
        self.input_channels = 3


class Simulation:
    def __init__(self, Config: CFG):
        self.Config = Config

    def add_white_noise(self, single_gyro_data):
        noise = np.random.normal(0.0, self.Config.noise_std, (self.Config.num_of_axis, self.Config.num_of_samples))
        return single_gyro_data + noise

    def add_bias(self, single_gyro_data, bias):
        return single_gyro_data + bias

    def create_single_gyro_data(self):
        bias_rand = np.random.normal(self.Config.bias, self.Config.bias_std, self.Config.num_of_axis)
        # bias_rand = self.Config.bias
        single_gyro_data = np.zeros([self.Config.num_of_axis, self.Config.num_of_samples])
        single_gyro_data = self.add_bias(single_gyro_data, bias_rand)
        single_gyro_data = self.add_white_noise(single_gyro_data)
        return single_gyro_data, bias_rand

    def create_dataset(self):
        dataset = np.zeros([self.Config.N_to_simulate, self.Config.num_of_axis, self.Config.num_of_samples])
        bias_labels = np.zeros([self.Config.N_to_simulate, self.Config.num_of_axis])
        for i in range(self.Config.N_to_simulate):
            dataset[i], bias_labels[i] = self.create_single_gyro_data()
        return dataset, bias_labels


def create_data(seed, output_dir):
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = CFG()
    sim = Simulation(config)
    for i in range(config.IMU_to_simulate):
        x, y = sim.create_dataset()
        imu_folder = os.path.join(output_dir, f"IMU_{i + 1}")
        if not os.path.exists(imu_folder):
            os.mkdir(imu_folder)

        for iteration_index in range(config.N_to_simulate):
            samples_filename = f"iteration_{iteration_index + 1}_samples.csv"
            samples_filepath = os.path.join(imu_folder, samples_filename)
            samples_content = x[iteration_index]
            samples_df = pd.DataFrame(samples_content)
            samples_df.to_csv(samples_filepath, index=False)

        bias_filename = f"IMU_{i + 1}_bias.csv"
        bias_filepath = os.path.join(imu_folder, bias_filename)
        bias_content = y
        bias_df = pd.DataFrame(bias_content)
        bias_df.to_csv(bias_filepath, index=False)
