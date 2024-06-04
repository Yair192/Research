import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from utils import running_avg

seed = 111
np.random.seed(seed)


samples = 13000
# Define the paths to the four folders
folder1_path = "/home/ystolero/Documents/Research/SF_data/1"
folder2_path = "/home/ystolero/Documents/Research/SF_data/2"
folder3_path = "/home/ystolero/Documents/Research/SF_data/3"
folder4_path = "/home/ystolero/Documents/Research/SF_data/4"

folder_paths = [folder1_path, folder2_path, folder3_path, folder4_path]

# Define the columns you want to load from the CSV files
columns_to_load = ["gx (dps)", "gy (dps)", "gz (dps)"]  # Add or remove columns as needed

# Initialize an empty array to store the data
data_array = np.zeros((4, 100, samples, 3))  # Replace `samples` with the actual number of samples

# Iterate over each folder
for folder_idx, folder_path in enumerate(folder_paths):
    # Initialize an empty list to store data from current folder
    folder_data = []

    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Select only the desired columns
            selected_columns = df[columns_to_load]

            # Append the selected columns to the folder data list
            folder_data.append(selected_columns.values)

    # Convert the folder data list into a NumPy array and store it in the data array
    data_array[folder_idx] = np.array(folder_data)

# Now, data_array has shape (4, 100, 3, samples), containing the desired data from all CSV files

sf_1_x = data_array[0, :, :, 0]



ind_to_train = [0, 3, 5, 9, 18, 24, 25, 27, 28, 30, 36, 43, 44, 53, 66, 67, 75, 82, 83, 95, 97]

# Convert indexes list to a NumPy array
indexes_array = np.array(ind_to_train)

# Select specific samples from sf_1_x using fancy indexing
# x_to_train = sf_1_x
# y_to_train = np.mean(x_to_train, axis=1)



x_to_train = np.zeros([5, 100, 3, 13000])

for i in range(4):
    x_to_train[i, :, 0,:] = data_array[i, :, : , 0]
    x_to_train[i, :, 1,:] = data_array[i, :, :, 1]
    x_to_train[i, :, 2,:] = data_array[i, :, :, 2]

## Try to mean all the data into 1 axis
# arrays_to_use = np.stack(x_to_train, axis=0)
# arrays_to_use = arrays_to_use[np.newaxis, :]
# x_to_train = data_array
y_to_train = np.mean(x_to_train, axis=3)

# x_to_train = x_to_train[np.newaxis, :]
# y_to_train = y_to_train[np.newaxis, :]

### Simulate new data based on the exists

mean_all = np.mean(x_to_train, axis=3)
mean_all = np.mean(mean_all, axis=1)

std_all = np.std(x_to_train, axis=3)
std_all = np.mean(np.std(std_all, axis=1))

new_array = np.zeros([100, 3, 13000])
new_array_biases = np.zeros([100, 3])

sim_bias_x = np.random.uniform(low=np.min(mean_all[:, 0]), high=np.max(mean_all[:, 0]), size=(1))
sim_bias_y = np.random.uniform(low=np.min(mean_all[:, 1]), high=np.max(mean_all[:, 1]), size=(1))
sim_bias_z = np.random.uniform(low=np.min(mean_all[:, 2]), high=np.max(mean_all[:, 2]), size=(1))

for i in range(new_array.shape[0]):

    new_array_biases[i, 0] = sim_bias_x
    new_array_biases[i, 1] = sim_bias_y
    new_array_biases[i, 2] = sim_bias_z

    x_axis = np.random.normal(loc=sim_bias_x, scale=0.01, size=(13000))
    y_axis = np.random.normal(loc=sim_bias_y, scale=0.01, size=(13000))
    z_axis = np.random.normal(loc=sim_bias_z, scale=0.01, size=(13000))

    new_array[i, 0] = x_axis
    new_array[i, 1] = y_axis
    new_array[i, 2] = z_axis

##### END SIMULATION

x_to_train[4] = new_array
y_to_train[4] = new_array_biases

print(x_to_train.shape)
print(y_to_train.shape)
# Define the file path
file_path = "/home/ystolero/Documents/Research/data/x_data_sim.npy"
filt_path_bias = "/home/ystolero/Documents/Research/data/y_data_sim.npy"

# Save the array
np.save(file_path, x_to_train)
np.save(filt_path_bias, y_to_train)

