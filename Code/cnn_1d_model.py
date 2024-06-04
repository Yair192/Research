import os
import torch.nn as nn
import torch
import torchvision.models as models  # Import ResNet model
from data_generator import CFG

os.chdir('/home/ystolero/Documents/Research/Simulation/Code/')

Config = CFG()
## START Raise input data parameters

# Load N = 10
# Train for 10 seconds
# Window size of 5 seconds with step 5

learning_rate = 0.0001
batch_size = 64
epochs = 300


## END Raise input data parameters


# ## START Raise axis parameters
# learning_rate = 0.0001
# batch_size = 64
# epochs = 300
# lstm_units = 1
# dense_units = 1
# output_units = 1


## END Raise axis parameters

## START Multihead parameters
# learning_rate = 0.00001
# batch_size = 64
# epochs = 300
## END Multihead parameters





class CNN1D(nn.Module):
    def __init__(self, input_ch):
        super(CNN1D, self).__init__()

        slope = 0.1
        kernel = 30
        stride = 1
        self.model_m = nn.Sequential(
            nn.Conv1d(in_channels=input_ch, out_channels=6, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(negative_slope=slope),
            nn.MaxPool1d(kernel_size=5)
        )


        self.fc_1 = nn.Linear(318, 128)
        self.fc_2 = nn.Linear(128, input_ch)
        # self.fc_3 = nn.Linear(8, 1)


    def forward(self, x):
        batch = x.shape[0]
        x_mean = torch.mean(x, dim=2)
        # x_mean = torch.reshape(x_mean, (batch,1))
        # x = x.unsqueeze(1)
        x = self.model_m(x)
        x = torch.flatten(x, start_dim=1)
        # x = torch.cat((x, x_mean), dim=1)
        x = self.fc_1(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.fc_2(x)
        # x = torch.nn.functional.leaky_relu(x, 0.1)
        # x = self.fc_3(x)
        # x = x.squeeze()
        return x


class LSTMGyro(nn.Module):
    def __init__(self, input_size, lstm_units, dense_units, output_units):
        super(LSTMGyro, self).__init__()
        # self.embedding = nn.EmbeddingBag(input_size, embedding_dim=32, sparse=True)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_units, batch_first=True)
        self.dense = nn.Linear(lstm_units, dense_units)
        self.output_layer = nn.Linear(dense_units, output_units)

    def forward(self, x):
        # Assuming x is a tensor with shape (batch_size, input_channels, num_samples)

        # LSTM layer

        x = x.unsqueeze(1)
        lstm_output, _ = self.lstm(x.transpose(1, 2))  # Transpose to (batch_size, num_samples, input_channels)

        # Dense layer
        dense_output = self.dense(lstm_output[:, -1, :])  # Use the last timestep's output

        # Output layer
        output = self.output_layer(dense_output)
        output = torch.squeeze(output, dim=1)
        return output


class CNN1DRaiseInput(nn.Module):
    def __init__(self, input_ch):
        super(CNN1DRaiseInput, self).__init__()
        self.input_ch = input_ch
        slope = 0.1
        kernel = 30
        stride = 1
        self.model_m = nn.Sequential(
            # nn.Conv1d(in_channels=input_ch, out_channels=12, kernel_size=kernel, stride=stride),
            # nn.LeakyReLU(negative_slope=slope),
            # nn.MaxPool1d(kernel_size=5),
            # nn.Conv1d(in_channels=12, out_channels=15, kernel_size=kernel, stride=stride),
            # nn.LeakyReLU(negative_slope=slope),
            # nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(in_channels=input_ch, out_channels=6, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(negative_slope=slope),
            nn.MaxPool1d(kernel_size=5)
        )


        # self.fc_1 = nn.Linear(60, 32)
        # self.fc_2 = nn.Linear(32, 16)
        # self.fc_3 = nn.Linear(16, input_ch)

        self.fc_1 = nn.Linear(318, 128)
        self.fc_2 = nn.Linear(128, input_ch)


    def forward(self, x):
        # x = self.model_m(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.fc_1(x)
        # x = torch.nn.functional.leaky_relu(x, 0.1)
        # x = self.fc_2(x)
        # x = torch.nn.functional.leaky_relu(x, 0.1)
        # x = self.fc_3(x)


        x = self.model_m(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_1(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.fc_2(x)
        return x


class MultiHeadCNN(nn.Module):
    def __init__(self, input_ch):
        super(MultiHeadCNN, self).__init__()

        self.input_ch = input_ch
        self.model = nn.ModuleList()
        slope = 0.1
        kernel = 30
        stride = 1
        for i in range(0,input_ch,3):
            self.model.append(nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=6, kernel_size=kernel, stride=stride),
                nn.LeakyReLU(negative_slope=slope),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=6, out_channels=9, kernel_size=kernel, stride=stride),
                nn.LeakyReLU(negative_slope=slope),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=9, out_channels=12, kernel_size=kernel, stride=stride),
                nn.LeakyReLU(negative_slope=slope),
                nn.MaxPool1d(kernel_size=2)
            ))

        # self.conv = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=3, kernel_size=kernel, stride=stride),
        #     nn.LeakyReLU(negative_slope=slope),
        #     nn.Conv1d(in_channels=3, out_channels=6, kernel_size=kernel, stride=stride),
        #     nn.LeakyReLU(negative_slope=slope)
        # )

        # # Calculate the number of features after convolution layers
        # self.num_features = self._calculate_num_features(input_ch)

        self.fc_1 = nn.Linear(132*int(self.input_ch/3), 128)
        self.fc_2 = nn.Linear(128, self.input_ch)

    # def _calculate_num_features(self, input_ch):
    #     # Create a temporary tensor to get the output shape after convolution layers
    #     with torch.no_grad():
    #         temp_tensor = torch.zeros(1, 1, Config.window_size)
    #         temp_output = self.conv(temp_tensor)
    #         num_features = temp_output.view(temp_output.size(0), -1).shape[1]
    #     return num_features * input_ch

    def forward(self, x):
        tensors_list = []
        for index in range(0, self.input_ch, 3):
            tensors_list.append(self.model[int(index/3)](x[:, index: index + 3, :]))
        conv_heads = torch.stack(tensors_list, dim=1)
        conv_heads = conv_heads.view(conv_heads.size(0), -1)
        # Forward through the fully connected layer
        fc_output = self.fc_1(conv_heads)
        fc_output = torch.nn.functional.leaky_relu(fc_output, 0.1)
        fc_output = self.fc_2(fc_output)
        return fc_output
