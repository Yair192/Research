import data_generator as dg
import os



# os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
output_directory = '/home/ystolero/Documents/Sim_data/train_like_real'
seed = 211

if not os.path.exists(output_directory):
    os.mkdir(output_directory)
dg.create_data(seed, output_directory)