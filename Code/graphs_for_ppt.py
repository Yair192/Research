import os
os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
import time
import numpy as np
import random
from matplotlib import pyplot as plt
from data_generator import CFG

Config = CFG()
x_train_all = np.load("/home/ystolero/Documents/Research/data/x_data_sim.npy") * Config.deg_s_to_rad_s

for i in range(100):
    data = x_train_all[0, i, 1, :]
    ra = []
    for i in range(data.shape[0]):
        ra.append(np.mean(data[:i]))

    ra = np.array(ra)

    plt.plot(Config.t_arr, np.abs(ra))
    plt.xlabel('Time [sec]')
    plt.ylabel('Running average [deg/s]')
    plt.show()


print(data.shape)

