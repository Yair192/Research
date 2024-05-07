import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cnn_1d_model as CNNet
import torchvision.models as models  # Import ResNet model
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from data_generator import CFG
from utils import create_segments_and_labels

seed = 111
np.random.seed(seed)
torch.manual_seed(seed)
os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')
Config = CFG()
x_train_all = np.load('/home/ystolero/Documents/Research/bias_data_from_slam_course_dots/data_array.npy')
y_train_all = np.mean(x_train_all, axis=3)
for k in range(Config.runs):
    # x_train_merged = np.zeros((0, Config.num_of_windows_train, Config.window_size))
    # y_train_merged = np.zeros((0, Config.num_of_windows_train))

    x_train_merged = np.zeros([0, Config.input_channels , Config.window_size])
    y_train_merged = np.zeros([0, Config.input_channels])
    RMSE_list = []
    for i in range(Config.IMU_to_train):
        print(f"Run Number: {k + 1} IMU Number: {i + 1}")
        # Get the IMU data
        if i == Config.test_imu_ind:
            X, y = x_train_all[i, :Config.records_to_train, :, 0:Config.samples_to_train], y_train_all[i, :Config.records_to_train]

        else:
            X, y = x_train_all[i, :, :, 0:Config.samples_to_train], y_train_all[i, :]

        # X_win, y_win = create_segments_and_labels(np.squeeze(X, axis=1), np.squeeze(y, axis=1), Config.window_size,
        #                                           Config.step_for_train, Config.ratio_window_to_step_for_train)

        # X_win = np.expand_dims(X_win, axis=1)
        # y_win = np.expand_dims(y_win, axis=1)
        x_train_merged = np.concatenate((x_train_merged, X), axis=0)
        y_train_merged = np.concatenate((y_train_merged, y), axis=0)

        # Reset the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNNet.CNN1D(Config.input_channels).to(device)
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        # criterion = CustomLoss(lambda_1, lambda_2)


        optimizer = optim.Adam(model.parameters(), lr=CNNet.learning_rate)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.01)
        model.to(device)


        # Data loaders
        # X_train, X_val, y_train, y_val = train_test_split(x_train_merged, y_train_merged, test_size=0.2,
        #                                                   random_state=seed)
        X_train = torch.Tensor(x_train_merged)
        y_train = torch.Tensor(y_train_merged)
        # X_val = torch.Tensor(X_val)
        # y_val = torch.Tensor(y_val)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=CNNet.batch_size, shuffle=True)
        # val_dataset = TensorDataset(X_val, y_val)
        # val_loader = DataLoader(val_dataset, batch_size=CNNet.batch_size, shuffle=False)

        # Train loop

        train_losses = []
        # val_losses = []
        # Training loop
        # min_val = 100000000
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
                # optimizer.zero_grad()  # zero the parameter gradients
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters

                # print statistics
                running_loss += loss.data.item()

            # StepLR scheduler step (update the learning rate)
            # scheduler.step()

            # Normalizing the loss by the total number of train batches
            running_loss /= len(train_loader)
            train_losses.append(running_loss)

            # Validation
            # model.eval()
            # val_loss = 0.0
            # total = 0

            # with torch.no_grad():
            #     for j, data in enumerate(val_loader, 0):
            #         inputs, labels = data
            #         inputs = inputs.to(device)
            #         labels = labels.to(device)
            #         outputs = model(inputs)
            #         val_loss += criterion(outputs, labels).data.item()
            #
            # val_loss /= len(val_loader)
            # val_losses.append(val_loss)
            log = "Epoch: {} | Train Loss: {:.12f} |".format(epoch, running_loss)
            # log = "Epoch: {} | Train Loss: {:.12f} | Val Loss: {:.12f} |".format(epoch, running_loss, val_loss)
            epoch_time = time.time() - epoch_time
            log += "Epoch Time: {:.2f} secs".format(epoch_time)
            print(log)

            # save model
            if epoch == 4:
            # if val_loss < min_val:
            #     min_val = val_loss
                print('==> Saving model ...')
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                if not os.path.isdir(f'/home/ystolero/Documents/Research/Simulation/checkpoints/run_{k}'):
                    os.mkdir(f'/home/ystolero/Documents/Research/Simulation/checkpoints/run_{k}')
                torch.save(state,
                           f'//home/ystolero/Documents/Research/Simulation/checkpoints/run_{k}/1d_cnn_ckpt_{i}.pth')
        print('==> Finished Training ...')

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
