import data_generator as dg
import os

os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
father_dir = '/home/ystolero/Documents/Sim_data'
output_directory = '/home/ystolero/Documents/Sim_data/test'
seed = 311

if not os.path.exists(father_dir):
    os.mkdir(father_dir)

if not os.path.exists(output_directory):
    os.mkdir(output_directory)
dg.create_data(seed, output_directory)