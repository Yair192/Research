from data_generator import CFG
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Config = CFG()

check_path = '/home/ystolero/Documents/Research/Real_data_30_min/8_D422CD004164_20240104_123927.csv'
columns_to_read = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
data = pd.read_csv(check_path, header=1, usecols=columns_to_read)
data = data.dropna()
data = data.to_numpy(dtype=float)
### Check samples for model based
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
x_avg = []
y_avg = []
z_avg = []
len = x.shape[0]
for k in range(120,len,120):
    x_to_avg = x[0:k]
    y_to_avg = y[0:k]
    z_to_avg = z[0:k]
    x_avg.append(np.mean(x_to_avg))
    y_avg.append(np.mean(y_to_avg))
    z_avg.append(np.mean(z_to_avg))
x_avg = np.array(x_avg)
y_avg = np.array(y_avg)
z_avg = np.array(z_avg)
# plt.plot(x_avg, label='X')
# plt.plot(y_avg, label='Y')
# plt.plot(z_avg, label='Z')
# plt.legend()
# plt.show()
plt.plot(x, label='X')
plt.plot(y, label='Y')
plt.plot(z, label='Z')
plt.legend()
plt.show()

# x_test_all = np.load('x_test.npy') * Config.deg_s_to_deg_h
# y_test_all = np.load('y_test.npy') * Config.deg_s_to_deg_h
# print(x_test_all.shape)
# print(y_test_all.shape)
#
# sample = x_test_all[0,0,0]
# bias = y_test_all[0,0]

# print(sample.shape)
# print(bias.shape)

# sample_sec = np.zeros(120)
#
# k = 0
# for i in range(120):
#     sample_sec[i] = np.mean(sample[0 : k])
#     k += Config.f
#
# plt.plot(sample_sec)
# plt.show()
#
# x_model_based = np.mean(sample) * Config.deg_s_to_deg_h
# y_model_based = 10
# error = y_model_based - x_model_based
# # print(error)
#
# error_array = []
# for j in range(120):
#     error_array.append(sample_sec[j] - y_model_based)
# error_array = np.array(error_array)
# plt.plot(error_array)
# plt.show()
#


