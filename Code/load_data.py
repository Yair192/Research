from utils import load_data as ld
from data_generator import CFG
import numpy as np

Config = CFG()

train_path = '/home/ystolero/Documents/Research/Simulation/data/train/'
test_path = '/home/ystolero/Documents/Research/Simulation/data/test/'

x_train, y_train = ld(Config.IMU_to_simulate, Config.N, Config.num_of_samples,
                             Config.num_of_axis, train_path)
x_test, y_test = ld(Config.IMU_to_test, Config.N, Config.num_of_samples,
                           Config.num_of_axis, test_path)

np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)