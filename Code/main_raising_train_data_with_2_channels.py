import os
import time
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

seed = 111
np.random.seed(seed)
torch.manual_seed(seed)
from sklearn.model_selection import train_test_split
from data_generator import CFG
import cnn_1d_model as CNNet

Config = CFG()

x_train_all = np.load('x_train.npy')
y_train_all = np.load('y_train.npy')

print("The X train all shape is:", x_train_all.shape)
print("The y train all shape is:", y_train_all.shape)

for k in range(Config.runs):
    x_train_merged = np.zeros((0, Config.input_channels + 1, Config.samples_to_train))
    y_train_merged = np.zeros((0, Config.input_channels + 1))
    RMSE_list = []
    print("Run Number:", k + 1)
    for i in range(0, Config.IMU_to_train, 2):
        print(f'IMU {i + 1} and IMU {i + 2}')
        # Get the IMU data
        X_1, y_1 = x_train_all[i, :, :, 0:Config.samples_to_train], y_train_all[i]
        X_2, y_2 = x_train_all[i + 1, :, :, 0:Config.samples_to_train], y_train_all[i + 1]
        X_2_IMU_merged = np.concatenate((X_1, X_2), axis=1)
        y_2_IMU_merged = np.concatenate((y_1, y_2), axis=1)
        x_train_merged = np.concatenate((x_train_merged, X_2_IMU_merged), axis=0)
        y_train_merged = np.concatenate((y_train_merged, y_2_IMU_merged), axis=0)

        # Reset the model

        model = CNNet.CNN1D(Config.input_channels + 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=CNNet.learning_rate)
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
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).data.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            log = "Epoch: {} | Train Loss: {:.12f} | Val Loss: {:.12f} |".format(epoch, running_loss, val_loss)
            epoch_time = time.time() - epoch_time
            log += "Epoch Time: {:.2f} secs".format(epoch_time)
            print(log)

            # save model
            if epoch == 4:
                print('==> Saving model ...')
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                if not os.path.isdir(f'/home/ystolero/Documents/Research/Simulation/checkpoint_2_ch_raise_data/run_{k}'):
                    os.mkdir(f'/home/ystolero/Documents/Research/Simulation/checkpoint_2_ch_raise_data/run_{k}')
                torch.save(state,
                           f'//home/ystolero/Documents/Research/Simulation/checkpoint_2_ch_raise_data/run_{k}/1d_cnn_ckpt_{i}.pth')
        print('==> Finished Training ...')

        # plot the loss curves
        x_epoch = list(range(1, CNNet.epochs + 1))
        plt.figure(figsize=(20, 6))  # Specify width and height in inches
        plt.plot(x_epoch, train_losses, label='Training  loss')
        plt.plot(x_epoch, val_losses, label='Validation  loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks(x_epoch)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
