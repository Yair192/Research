import os
os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
import time
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import cnn_1d_model as CNNet
import torchvision.models as models  # Import ResNet model
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from data_generator import CFG
from utils import create_seg_for_single_gyro

seed = 111
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

Config = CFG()

x_train_all = np.load("/home/ystolero/Documents/Research/data/x_data_sim.npy") * Config.deg_s_to_rad_s
y_train_all = np.load("/home/ystolero/Documents/Research/data/y_data_sim.npy") * Config.deg_s_to_rad_s

best_epoch_for_imu = np.zeros(Config.IMU_to_train)
best_loss_for_imu = np.zeros(Config.IMU_to_train)

# create a nn class (just-for-fun choice :-)

class CustomLoss(nn.Module):
    def __init__(self, lambda_1, lambda_2):
        super(CustomLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, pred, label, mean):
        mse_loss = nn.MSELoss()
        mse_pred = mse_loss(pred, label)
        mse_mean = mse_loss(mean, label)
        loss = self.lambda_1 * mse_pred + self.lambda_2 * mse_mean
        return loss

# Define lambdas
lambda_1 = 1
lambda_2 = 1.5


for k in range(Config.runs):


    x_train_merged = np.zeros([Config.num_of_windows_train*Config.records_to_train, 0, Config.window_size])
    y_train_merged = np.zeros([Config.num_of_windows_train*Config.records_to_train, 0])
    RMSE_list = []
    records_index = 0
    for i in range(Config.IMU_to_train):
        print(f"Run Number: {k + 1} IMU Number: {i + 1}")
        # Get the IMU data'


        X, y = x_train_all[i, :Config.records_to_train, :, 0:Config.samples_to_train], y_train_all[i, :Config.records_to_train]

        X_win, y_win = create_seg_for_single_gyro(X, y, Config.window_size, Config.step_for_train)


        x_train_merged = np.concatenate((x_train_merged, X_win), axis=1)
        y_train_merged = np.concatenate((y_train_merged, y_win), axis=1)

        x_mean_x = np.mean(x_train_merged[:, :x_train_merged.shape[1]:3], axis=1)
        x_mean_y = np.mean(x_train_merged[:, 1:x_train_merged.shape[1]:3], axis=1)
        x_mean_z = np.mean(x_train_merged[:, 2:x_train_merged.shape[1]:3], axis=1)

        y_mean_x = np.mean(y_train_merged[:, :x_train_merged.shape[1]:3], axis=1)
        y_mean_y = np.mean(y_train_merged[:, 1:x_train_merged.shape[1]:3], axis=1)
        y_mean_z = np.mean(y_train_merged[:, 2:x_train_merged.shape[1]:3], axis=1)

        x_train_merged = np.stack((x_mean_x, x_mean_y, x_mean_z), axis=1)
        y_train_merged = np.stack((y_mean_x, y_mean_y, y_mean_z), axis=1)
        # Reset the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNNet.CNN1D(Config.input_channels).to(device)
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        # criterion = CustomLoss(lambda_1, lambda_2)


        optimizer = optim.Adam(model.parameters(), lr=CNNet.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        X_train = torch.Tensor(x_train_merged)
        y_train = torch.Tensor(y_train_merged)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=CNNet.batch_size, shuffle=True)


        # Train loop

        train_losses = []

        # Training loop

        min_train = 100000000
        for epoch in range(CNNet.epochs):
            model.train()
            running_loss = 0.0
            epoch_time = time.time()

            for j, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data
                # send them to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # forward + backward + optimize
                mean = torch.mean(inputs, dim=2)
                outputs = model(inputs)  # forward pass
                loss = criterion(outputs, labels)  # calculate the loss

                # always the same 3 steps
                optimizer.zero_grad()  # zero the parameter gradients
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters

                # print statistics
                running_loss += loss.data.item()

            # StepLR scheduler step (update the learning rate)
            scheduler.step()
            if epoch == 200:
                print(optimizer.param_groups[0]['lr'])

            # Normalizing the loss by the total number of train batches
            running_loss /= len(train_loader)
            train_losses.append(running_loss)

            # Validation

            if epoch % 10 == 0:
                log = "Epoch: {} | Train Loss: {:.12f} |".format(epoch, running_loss)
                # log = "Epoch: {} | Train Loss: {:.12f} | Val Loss: {:.12f} |".format(epoch, running_loss, val_loss)
                epoch_time = time.time() - epoch_time
                log += "Epoch Time: {:.2f} secs".format(epoch_time)
                print(log)

            # save model

            if running_loss < min_train:
                min_train = running_loss
                best_epoch_for_imu[i] = epoch
                best_loss_for_imu[i] = min_train
                print('==> Saving model ...')
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                if not os.path.isdir(f'/home/ystolero/Documents/Research/checkpoints/run_{k}'):
                    os.mkdir(f'/home/ystolero/Documents/Research/checkpoints/run_{k}')
                torch.save(state,
                           f'//home/ystolero/Documents/Research/checkpoints/run_{k}/1d_cnn_ckpt_{i}.pth')
        print('==> Finished Training ...')


print(f"The best epoch per IMU is: {best_epoch_for_imu}")
print(f"The best loss per IMU is: {best_loss_for_imu}")
        #
        # # plot the loss curves
        # x_epoch = list(range(1, CNNet.epochs + 1))
        # plt.figure(figsize=(20, 6))  # Specify width and height in inches
        # plt.plot(x_epoch, train_losses, label='Training  loss')
        # # plt.plot(x_epoch, val_losses, label='Validation  loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.xticks(x_epoch)
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
