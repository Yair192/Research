import data_generator as dg
import os

output_directory = '/home/ystolero/Documents/Research/Simulation/data/train'
seed = 211

if not os.path.exists(output_directory):
    os.mkdir(output_directory)
dg.create_data(seed, output_directory)