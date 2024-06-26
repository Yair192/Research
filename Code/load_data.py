import os
import numpy as np
from utils import load_data as ld
from data_generator import CFG


Config = CFG()

os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
train_path = '/home/ystolero/Documents/Sim_data/train/'
test_path = '/home/ystolero/Documents/Sim_data/test/'
model_based_path = '/home/ystolero/Documents/Sim_data/model_based/'

x_train, y_train = ld(Config.IMU_to_simulate, Config.N_to_load, Config.num_of_samples,
                             Config.num_of_axis, train_path)
x_test, y_test = ld(Config.IMU_to_test, Config.N_to_load, Config.num_of_samples,
                           Config.num_of_axis, test_path)


np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)