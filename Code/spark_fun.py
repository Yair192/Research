import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import running_avg


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

# x_to_train = np.zeros([4, 100, 3, 13000])

# j = 0
# for i in range(x_to_train.shape[0]):
#     x_to_train[i, :, :] = data_array[j, :, : , 0]
#     x_to_train[i+1, :, :] = data_array[j, :, :, 1]
#     x_to_train[i+2, :, :] = data_array[j, :, :, 2]
#     j += 1

## Try to mean all the data into 1 axis
# arrays_to_use = np.stack(x_to_train, axis=0)
# arrays_to_use = arrays_to_use[np.newaxis, :]
x_to_train = data_array
y_to_train = np.mean(x_to_train, axis=2)

# x_to_train = x_to_train[np.newaxis, :]
# y_to_train = y_to_train[np.newaxis, :]

print(x_to_train.shape)
print(y_to_train.shape)
# Define the file path
file_path = "/home/ystolero/Documents/Research/data/x_data_sim.npy"
filt_path_bias = "/home/ystolero/Documents/Research/data/y_data_sim.npy"

# Save the array
np.save(file_path, x_to_train)
np.save(filt_path_bias, y_to_train)
