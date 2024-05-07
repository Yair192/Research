import torch.nn as nn
import torch
import torchvision.models as models  # Import ResNet model
from data_generator import CFG




Config = CFG()
## START Raise input data parameters

# Load N = 10
# Train for 10 seconds
# Window size of 5 seconds with step 5


learning_rate = 0.00001
batch_size = 8
epochs = 300

## END Raise input data parameters


## START Raise axis parameters
# learning_rate = 0.5
# batch_size = 4
# epochs = 100כאי
# lstm_units = 4
# dense_units = 4
# output_units = 1


## END Raise axis parameters

## START Multihead parameters
# learning_rate = 0.00000005
# batch_size = 8
# epochs = 50
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
            # nn.Conv1d(in_channels=3, out_channels=6, kernel_size=kernel, stride=stride),
            # nn.LeakyReLU(negative_slope=slope)
        )


        self.fc_1 = nn.Linear(156, 64)
        self.fc_2 = nn.Linear(64, input_ch)
        # self.fc_3 = nn.Linear(8, 1)


    def forward(self, x):
        batch = x.shape[0]
        x_mean = torch.mean(x, dim=2)
        # x_mean = torch.reshape(x_mean, (batch,1))
        # x = x.unsqueeze(1)


        x = self.fc_2(x)

        # x = torch.nn.functional.leaky_relu(x, 0.1)
        # x = torch.nn.functional.tanh(x)
        # x = self.fc_3(x)

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
        lstm_output, _ = self.lstm(x.transpose(1, 2))  # Transpose to (batch_size, num_samples, input_channels)

        # Dense layer
        dense_output = self.dense(lstm_output[:, -1, :])  # Use the last timestep's output

        # Output layer
        output = self.output_layer(dense_output)

        return output


class CNN1DRaiseInput(nn.Module):
    def __init__(self, input_ch):
        super(CNN1DRaiseInput, self).__init__()
        self.input_ch = input_ch
        slope = 0.5
        kernel = (self.input_ch, 5)
        stride = 2
        pad = self.input_ch + 10
        self.model_m = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(negative_slope=slope),
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=3, kernel_size=kernel, stride=stride, padding=pad),
            nn.LeakyReLU(negative_slope=slope)
        )

        # Calculate the number of features after convolution layers
        self.num_features = self._calculate_num_features()

        self.fc_1 = nn.Linear(self.num_features, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, self.input_ch)

    def _calculate_num_features(self):
        # Create a temporary tensor to get the output shape after convolution layers
        with torch.no_grad():
            temp_tensor = torch.zeros(1, 1, self.input_ch, Config.window_size)
            temp_output = self.model_m(temp_tensor)
            num_features = temp_output.view(temp_output.size(0), -1).shape[1]
        return num_features

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model_m(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.fc_2(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.fc_3(x)
        return x


class MultiHeadCNN(nn.Module):
    def __init__(self, input_ch):
        super(MultiHeadCNN, self).__init__()

        self.input_ch = input_ch
        self.model = nn.ModuleList()
        slope = 0.1
        kernel = 5
        stride = 1
        for i in range(input_ch):
            self.model.append(nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=3, kernel_size=kernel, stride=stride),
                nn.LeakyReLU(negative_slope=slope),
                nn.Conv1d(in_channels=3, out_channels=6, kernel_size=kernel, stride=stride),
                nn.LeakyReLU(negative_slope=slope)
            ))

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(negative_slope=slope),
            nn.Conv1d(in_channels=3, out_channels=6, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(negative_slope=slope)
        )

        # Calculate the number of features after convolution layers
        self.num_features = self._calculate_num_features(input_ch)

        self.fc_1 = nn.Linear(self.num_features, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 1)

    def _calculate_num_features(self, input_ch):
        # Create a temporary tensor to get the output shape after convolution layers
        with torch.no_grad():
            temp_tensor = torch.zeros(1, 1, Config.window_size)
            temp_output = self.conv(temp_tensor)
            num_features = temp_output.view(temp_output.size(0), -1).shape[1]
        return num_features * input_ch

    def forward(self, x):
        tensors_list = []
        for index in range(0, self.input_ch):
            tensors_list.append(self.model[index](x[:, index: index + 1, :]))
        conv_heads = torch.stack(tensors_list, dim=1)
        conv_heads = conv_heads.view(conv_heads.size(0), -1)
        # Forward through the fully connected layer
        fc_output = self.fc_1(conv_heads)
        fc_output = torch.nn.functional.leaky_relu(fc_output, 0.1)
        fc_output = self.fc_2(fc_output)
        fc_output = torch.nn.functional.leaky_relu(fc_output, 0.1)
        fc_output = self.fc_3(fc_output)
        fc_output = torch.nn.functional.leaky_relu(fc_output, 0.1)
        return fc_output

class ResNet1D(nn.Module):
    def __init__(self, input_channels: int = 1, output_features: int = 1, drop_out: float = 0.5):
        super(ResNet1D, self).__init__()

        self.input_channels = input_channels
        self.output_features = output_features

        self.model = models.resnet18()
        expansion = 1  # 4 for resnet50

        self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512*expansion, self.output_features)

        self.dp1 = nn.Dropout(drop_out)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        return self.model(x)

# class ResNet1D(nn.Module):
#     def __init__(self, input_channels=1, num_classes=1):
#         super(ResNet1D, self).__init__()
#         self.resnet = models.resnet18(pretrained=False)  # Load ResNet-18
#         # Replace the first convolutional layer to adapt for 1D data
#         self.resnet.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # Replace the fully connected layer for regression
#         self.resnet.fc = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         # Ensure input is 3D (batch size, channels, sequence length)
#         x = x.unsqueeze(2)  # Add dummy height dimension
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)
#
#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)
#
#         x = F.adaptive_avg_pool1d(x, 1)
#         x = torch.flatten(x, 1)
#         x = self.resnet.fc(x)
#         return x