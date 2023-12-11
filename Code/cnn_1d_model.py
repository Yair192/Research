import torch.nn as nn
import torch
from data_generator import CFG

Config = CFG()
learning_rate = 0.00001
batch_size = 2
epochs = 5


class CNN1D(nn.Module):
    def __init__(self, input_ch):
        super(CNN1D, self).__init__()

        self.model_m = nn.Sequential(
            nn.Conv1d(in_channels=input_ch, out_channels=8, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )

        # Calculate the number of features after convolution layers
        self.num_features = self._calculate_num_features(input_ch)

        self.fc = nn.Linear(self.num_features, input_ch)

    def _calculate_num_features(self, input_ch):
        # Create a temporary tensor to get the output shape after convolution layers
        with torch.no_grad():
            temp_tensor = torch.zeros(1, input_ch, Config.num_of_samples)
            temp_output = self.model_m(temp_tensor)
            num_features = temp_output.view(temp_output.size(0), -1).shape[1]
        return num_features

    def forward(self, x, input_ch):
        x = self.model_m(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
