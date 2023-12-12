import matplotlib.pyplot as plt
import os
from data_generator import CFG
import numpy as np
import pandas as pd

Config = CFG()

os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
x_train_all = np.load('/home/ystolero/Documents/Research/bias_data_from_slam_course_dots/data_array.npy')
y_train_all = np.mean(x_train_all, axis=3)
x_train_all = np.squeeze(x_train_all, axis=2)
y_train_all = np.squeeze(y_train_all, axis=2)

y_train_merged = np.zeros(0)


y_test_0 = y_train_all[0,90:]
# y_train_0 = np.concatenate((y_train_merged, y_train_all[0,:90]), axis=0)
# y_train_1 = np.concatenate((y_train_0, y_train_all[1]), axis=0)
# y_train_2 = np.concatenate((y_train_1, y_train_all[2]), axis=0)
# y_train_3 = np.concatenate((y_train_2, y_train_all[3]), axis=0)
# y_train_4 = np.concatenate((y_train_3, y_train_all[4]), axis=0)
# y_train_5 = np.concatenate((y_train_4, y_train_all[5]), axis=0)
# y_train_6 = np.concatenate((y_train_5, y_train_all[6]), axis=0)
# y_train_7 = np.concatenate((y_train_6, y_train_all[7]), axis=0)
# y_train_8 = np.concatenate((y_train_7, y_train_all[8]), axis=0)
# y_train_9 = np.concatenate((y_train_8, y_train_all[9]), axis=0)

y_train_0 = y_train_all[0,:90]
y_train_1 = y_train_all[1]
y_train_2 = y_train_all[2]
y_train_3 = y_train_all[3]
y_train_4 = y_train_all[4]
y_train_5 = y_train_all[5]
y_train_6 = y_train_all[6]
y_train_7 = y_train_all[7]
y_train_8 = y_train_all[8]
y_train_9 = y_train_all[9]

x_train_merged = np.zeros((0, Config.window_size))
#
# x_test_0 = x_train_all[0, 90:, 0:Config.samples_to_train].flatten()
# x_train_0 = np.concatenate((x_train_merged, x_train_all[0, :90, 0:Config.samples_to_train]), axis=0).flatten()
# x_train_1 = np.concatenate((x_train_0, x_train_all[1, :, 0:Config.samples_to_train].flatten()), axis=0)
# x_train_2 = np.concatenate((x_train_1, x_train_all[2, :, 0:Config.samples_to_train].flatten()), axis=0)
# x_train_3 = np.concatenate((x_train_2, x_train_all[3, :, 0:Config.samples_to_train].flatten()), axis=0)
# x_train_4 = np.concatenate((x_train_3, x_train_all[4, :, 0:Config.samples_to_train].flatten()), axis=0)
# x_train_5 = np.concatenate((x_train_4, x_train_all[5, :, 0:Config.samples_to_train].flatten()), axis=0)
# x_train_6 = np.concatenate((x_train_5, x_train_all[6, :, 0:Config.samples_to_train].flatten()), axis=0)
# x_train_7 = np.concatenate((x_train_6, x_train_all[7, :, 0:Config.samples_to_train].flatten()), axis=0)
# x_train_8 = np.concatenate((x_train_7, x_train_all[8, :, 0:Config.samples_to_train].flatten()), axis=0)
# x_train_9 = np.concatenate((x_train_8, x_train_all[9, :, 0:Config.samples_to_train].flatten()), axis=0)

x_train_0 = x_train_all[0, :90].flatten()
x_train_1 = x_train_all[1].flatten()
x_train_2 = x_train_all[2].flatten()
x_train_3 = x_train_all[3].flatten()
x_train_4 = x_train_all[4].flatten()
x_train_5 = x_train_all[5].flatten()
x_train_6 = x_train_all[6].flatten()
x_train_7 = x_train_all[7].flatten()
x_train_8 = x_train_all[8].flatten()
x_train_9 = x_train_all[9].flatten()

# Concatenate the arrays along the first axis (axis=0)
all_data = np.concatenate((x_train_2, x_train_3, x_train_6, x_train_9), axis=0)

mean_1 = np.mean(x_train_2)
mean_2 = np.mean(x_train_3)
mean_3 = np.mean(x_train_6)
mean_4 = np.mean(x_train_9)

min_mean = min(mean_1, mean_2, mean_3, mean_4)
max_mean = max(mean_1, mean_2, mean_3, mean_4)


# Calculate the mean and standard deviation along the first axis
std_dev = np.std(all_data, axis=0)

# simulated_data = np.random.normal(loc=mean, scale=std_dev, size=all_data.shape[0]).flatten()
#
# # Reshape the flattened data back to the original shape of the arrays
# reshaped_data = simulated_data.reshape(4, 100, -1)

records = 100

array1_new = np.zeros([records, 7200])
array2_new = np.zeros([records, 7200])
array3_new = np.zeros([records, 7200])
array4_new = np.zeros([records, 7200])
array5_new = np.zeros([records, 7200])
array6_new = np.zeros([records, 7200])
array7_new = np.zeros([records, 7200])
array8_new = np.zeros([records, 7200])


bias_array_1 = np.zeros(records)
bias_array_2 = np.zeros(records)
bias_array_3 = np.zeros(records)
bias_array_4 = np.zeros(records)
bias_array_5 = np.zeros(records)
bias_array_6 = np.zeros(records)
bias_array_7 = np.zeros(records)
bias_array_8 = np.zeros(records)

bias_array_1 = np.random.uniform(low=min_mean, high=max_mean, size=records)
bias_array_2 = np.random.uniform(low=min_mean, high=max_mean, size=records)
bias_array_3 = np.random.uniform(low=min_mean, high=max_mean, size=records)
bias_array_4 = np.random.uniform(low=min_mean, high=max_mean, size=records)
bias_array_5 = np.random.uniform(low=min_mean, high=max_mean, size=records)
bias_array_6 = np.random.uniform(low=min_mean, high=max_mean, size=records)
bias_array_7 = np.random.uniform(low=min_mean, high=max_mean, size=records)
bias_array_8 = np.random.uniform(low=min_mean, high=max_mean, size=records)

for i in range(records):
    array1_new[i] = np.random.normal(loc=bias_array_1[i], scale=std_dev, size=7200)
    array2_new[i] = np.random.normal(loc=bias_array_2[i], scale=std_dev, size=7200)
    array3_new[i] = np.random.normal(loc=bias_array_3[i], scale=std_dev, size=7200)
    array4_new[i] = np.random.normal(loc=bias_array_4[i], scale=std_dev, size=7200)
    array5_new[i] = np.random.normal(loc=bias_array_5[i], scale=std_dev, size=7200)
    array6_new[i] = np.random.normal(loc=bias_array_6[i], scale=std_dev, size=7200)
    array7_new[i] = np.random.normal(loc=bias_array_7[i], scale=std_dev, size=7200)
    array8_new[i] = np.random.normal(loc=bias_array_8[i], scale=std_dev, size=7200)
# Extract individual arrays
# array1_new = reshaped_data[0]
# array2_new = reshaped_data[1]
# array3_new = reshaped_data[2]
# array4_new = reshaped_data[3]
# Combine the arrays along the first axis
combined_data = np.stack((array1_new, array2_new, array3_new, array4_new, array5_new, array6_new, array7_new, array8_new), axis=0)
bias_data = np.stack((bias_array_1, bias_array_2, bias_array_3, bias_array_4, bias_array_5, bias_array_6, bias_array_7, bias_array_8), axis=0)


print(array1_new.shape)
print(array2_new.shape)
print(array3_new.shape)
print(array4_new.shape)
print(combined_data.shape)

# Define the file path
file_path = "x_data_sim.npy"
filt_path_bias = "y_data_sim.npy"

# Save the array
np.save(file_path, combined_data)
np.save(filt_path_bias, bias_data)



colors_rgb = [
    "#FF0000",  # Red
    "#00FF00",  # Green
    "#0000FF",  # Blue
    "#FFFF00",  # Yellow
    "#FF00FF",  # Magenta
    "#00FFFF",  # Cyan
    "#800000",  # Maroon
    "#008000",  # Green (dark)
    "#000080",  # Navy
    "#808000"   # Olive
]

# plt.hist(y_test_0, label='1st IMU - test only', bins=10)
# plt.legend()
# plt.show()
#
# plt.hist(y_train_0, label='1st IMU - train', bins=10, color=colors_rgb[1])
# plt.legend()
# # plt.show()
#
# plt.hist(y_train_1, label='2nd IMU - train', bins=10, color=colors_rgb[2])
# plt.legend()
# # plt.show()
#
# plt.hist(y_train_2, label='3nd IMU - train', bins=30, color=colors_rgb[3])
# plt.legend()
# # plt.show()
#
# plt.hist(y_train_3, label='4nd IMU - train', bins=30, color=colors_rgb[4])
# plt.legend()
# # plt.show()
#
# plt.hist(y_train_4, label='5nd IMU - train', bins=30, color=colors_rgb[5])
# plt.legend()
# # plt.show()
#
# plt.hist(y_train_5, label='6nd IMU - train', bins=30, color=colors_rgb[6])
# plt.legend()
# # plt.show()
#
# plt.hist(y_train_6, label='7nd IMU - train', bins=30, color=colors_rgb[7])
# plt.legend()
# # plt.show()
#
# plt.hist(y_train_7, label='8nd IMU - train', bins=30, color=colors_rgb[8])
# plt.legend()
# # plt.show()
#
# plt.hist(y_train_8, label='9nd IMU - train', bins=30, color=colors_rgb[9])
# plt.legend()
# # plt.show()
#
# plt.hist(y_train_9, label='10nd IMU - train', bins=100, color=colors_rgb[0])
# plt.legend()
# plt.show()




# plt.hist(x_test_0, label='1st IMU - test only', bins=1000)
# plt.legend()
# # plt.show()

plt.hist(x_train_0, label='1st IMU - train', bins=1000, color=colors_rgb[1])
plt.legend()
# plt.show()

plt.hist(x_train_1, label='2nd IMU - train', bins=1000, color=colors_rgb[2])
plt.legend()
# plt.show()

plt.hist(x_train_2, label='3nd IMU - train', bins=1000, color=colors_rgb[3])
plt.legend()
# plt.show()

plt.hist(x_train_3, label='4nd IMU - train', bins=1000, color=colors_rgb[4])
plt.legend()
# plt.show()

plt.hist(x_train_4, label='5nd IMU - train', bins=1000, color=colors_rgb[5])
plt.legend()
# plt.show()

plt.hist(x_train_5, label='6nd IMU - train', bins=1000, color=colors_rgb[6])
plt.legend()
# plt.show()

plt.hist(x_train_6, label='7nd IMU - train', bins=1000, color=colors_rgb[7])
plt.legend()
# plt.show()

plt.hist(x_train_7, label='8nd IMU - train', bins=1000, color=colors_rgb[8])
plt.legend()
# plt.show()

plt.hist(x_train_8, label='9nd IMU - train', bins=1000, color=colors_rgb[9])
plt.legend()
# plt.show()


# # Plot histogram with just the outline
# plt.hist(simulated_data, bins=1000, histtype='step', color='black', linewidth=1.5, label='Histogram of Simulated Data')

# Plot histogram with just the outline
plt.hist(array1_new.flatten(), bins=1000, histtype='step', color='red', linewidth=1.5, label='Array 1 sim')

# Plot histogram with just the outline
plt.hist(array2_new.flatten(), bins=1000, histtype='step', color='green', linewidth=1.5, label='Array 1 sim')

# Plot histogram with just the outline
plt.hist(array3_new.flatten(), bins=1000, histtype='step', color='blue', linewidth=1.5, label='Array 1 sim')

# Plot histogram with just the outline
plt.hist(array4_new.flatten(), bins=1000, histtype='step', color='yellow', linewidth=1.5, label='Array 1 sim')

plt.hist(x_train_9, label='10nd IMU - train', bins=1000, color=colors_rgb[0])
plt.legend()
plt.show()
