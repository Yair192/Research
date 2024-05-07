import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cnn_1d_model as CNNet
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from data_generator import CFG
from utils import create_segments_and_labels

seed = 111
np.random.seed(seed)
torch.manual_seed(seed)

Config = CFG()
os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
x_train_all = np.load('x_train.npy') * Config.deg_s_to_deg_h
y_train_all = np.load('y_train.npy') * Config.deg_s_to_deg_h
for k in range(Config.runs):
    x_train_merged = np.zeros((Config.num_of_windows, 0, Config.window_size))
    y_train_merged = np.zeros((Config.num_of_windows, 0))
    RMSE_list = []
    input_channels = 0
    for i in range(Config.IMU_to_train):

        print(f'Run number {k + 1} for IMU number {i + 1}')

        input_channels += 1

        # Get the IMU data
        X, y = x_train_all[i, :, :, 0:Config.samples_to_train], y_train_all[i]

        X_win, y_win = create_segments_and_labels(np.squeeze(X, axis=1), np.squeeze(y, axis=1), Config.window_size,
                                                  Config.step_for_train, Config.ratio_window_to_step_for_train)
        X_win = np.expand_dims(X_win, axis=1)
        y_win = np.expand_dims(y_win, axis=1)

        x_train_merged = np.concatenate((x_train_merged, X_win), axis=1)
        y_train_merged = np.concatenate((y_train_merged, y_win), axis=1)

        # Reset the model

        model = CNNet.MultiHeadCNN(input_channels)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=CNNet.learning_rate)
        # optimizer = optim.RMSprop(model.parameters(), lr=CNNet.learning_rate)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Data loaders
        X_train, X_val, y_train, y_val = train_test_split(x_train_merged, y_train_merged, test_size=0.2,
                                                          random_state=seed)
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train)
        X_val = torch.Tensor(X_val)
        y_val = torch.Tensor(y_val)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=CNNet.batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=CNNet.batch_size, shuffle=False)

        # Train loop

        train_losses = []
        val_losses = []
        # Training loop
        min_val = 100000000
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
                labels = torch.mean(labels, dim=1, keepdim=True)
                # forward + backward + optimize
                outputs = model(inputs)  # forward pass
                loss = criterion(outputs, labels)  # calculate the loss
                # always the same 3 steps
                optimizer.zero_grad()  # zero the parameter gradients
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters

                # print statistics
                running_loss += loss.data.item()

            # Normalizing the loss by the total number of train batches
            running_loss /= len(train_loader)
            train_losses.append(running_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            total = 0

            with torch.no_grad():
                for j, data in enumerate(val_loader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    labels = torch.mean(labels, dim=1, keepdim=True)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).data.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            log = "Epoch: {} | Train Loss: {:.12f} | Val Loss: {:.12f} |".format(epoch, running_loss, val_loss)
            epoch_time = time.time() - epoch_time
            log += "Epoch Time: {:.2f} secs".format(epoch_time)
            print(log)

            # save model
            if val_loss < min_val:
                min_val = val_loss
                print('==> Saving model ...')
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                if not os.path.isdir(f'/home/ystolero/Documents/Research/Simulation/checkpoints_multihead/run_{k}'):
                    os.mkdir(f'/home/ystolero/Documents/Research/Simulation/checkpoints_multihead/run_{k}')
                torch.save(state,
                           f'//home/ystolero/Documents/Research/Simulation/checkpoints_multihead/run_{k}/1d_cnn_ckpt_{i}.pth')
        print('==> Finished Training ...')

        # # plot the loss curves
        # x_epoch = list(range(1, CNNet.epochs + 1))
        # plt.figure(figsize=(20, 6))  # Specify width and height in inches
        # plt.plot(x_epoch, train_losses, label='Training  loss')
        # plt.plot(x_epoch, val_losses, label='Validation  loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.xticks(x_epoch)
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        # plt.close('all')