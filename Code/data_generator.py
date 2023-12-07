import os
import numpy as np
import pandas as pd
import torch


class CFG:
    def __init__(self):
        self.f = 200
        self.t = 30
        self.num_of_samples = self.f * self.t
        self.N = 50
        self.IMU_to_simulate = 16
        self.IMU_to_test = 1
        self.IMU_to_train = 16
        self.runs = 10
        self.deg_h_to_rad_s = (np.pi / 180) * (1 / 3600)
        self.rad_s_to_deg_s = 180 / np.pi
        self.bias = 10 * self.deg_h_to_rad_s
        self.num_of_axis = 1
        self.input_channels = 1


class Simulation:
    def __init__(self, Config: CFG):
        self.Config = Config

    def add_white_noise(self, single_gyro_data):
        noise_std = np.deg2rad(0.007) * np.sqrt(self.Config.f)
        noise = np.random.normal(0.0, noise_std, (self.Config.num_of_axis, self.Config.num_of_samples))
        return single_gyro_data + noise

    def add_bias(self, single_gyro_data):
        return single_gyro_data + self.Config.bias

    def create_single_gyro_data(self):
        bias_rand = np.random.normal(0.0, self.Config.bias, self.Config.num_of_axis)
        single_gyro_data = np.zeros([self.Config.num_of_axis, self.Config.num_of_samples])
        single_gyro_data = self.add_bias(single_gyro_data)
        single_gyro_data = self.add_white_noise(single_gyro_data)
        return single_gyro_data, bias_rand

    def create_dataset(self):
        dataset = np.zeros([self.Config.N, self.Config.num_of_axis, self.Config.num_of_samples])
        bias_labels = np.zeros([self.Config.N, self.Config.num_of_axis])
        for i in range(self.Config.N):
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

        for iteration_index in range(config.N):
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
