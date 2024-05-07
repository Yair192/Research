from data_generator import CFG
import numpy as np
import pandas as pd

Config = CFG()

# train_path_1 = '/home/ystolero/Documents/Research/RealData/8_D422CD004164_20230104_084015.csv'
# train_path_2 = '/home/ystolero/Documents/Research/RealData/10_D422CD003EBB_20230104_084015.csv'
test_path_1 = '/home/ystolero/Documents/Research/RealData/9_D422CD003E6A_20230104_084015.csv'
# test_path_2 = '/home/ystolero/Documents/Research/RealData/11_D422CD003E87_20230104_084015.csv'

columns_to_read = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']

# train_data_1 = pd.read_csv(train_path_1, header=1, usecols=columns_to_read)
# train_data_2 = pd.read_csv(train_path_2, header=1, usecols=columns_to_read)
test_data_1 = pd.read_csv(test_path_1, header=1, usecols=columns_to_read)
# test_data_2 = pd.read_csv(test_path_2, header=1, usecols=columns_to_read)

# train_data = pd.concat([train_data_1, train_data_2], axis=1)
# test_data = pd.concat([test_data_1, test_data_2], axis=1)

# train_data = train_data.dropna()
test_data = test_data_1.dropna()

# train_data = train_data.values.T
test_data = test_data.values.T

# num_windows = train_data.shape[1] // Config.samples_in_window

# windowed_train_data = train_data[:, :num_windows * Config.samples_in_window].reshape(
#     (6, num_windows, 1, Config.samples_in_window))

# windowed_test_data = test_data[:, :num_windows * Config.samples_in_window].reshape(
#     (6, num_windows, 1, Config.samples_in_window))
#
# train_labels = np.zeros([6, num_windows, 1])
test_labels = np.zeros([3, 1])
# for i in range(6):
    # train_labels[i] = np.mean(train_data[i])
# for i in range(3):
#     test_labels[i] = np.mean(test_data[i])


# np.save('x_train_real.npy', windowed_train_data)
# np.save('y_train_real.npy', train_labels)
np.save('x_test_real.npy', test_data)
# np.save('y_test_real.npy', test_labels)
