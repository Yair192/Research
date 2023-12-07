import torch.nn as nn


learning_rate = 0.00001
batch_size = 2
epochs = 5


class CNN1D(nn.Module):
    def __init__(self, input_channels):
        super(CNN1D, self).__init__()

        self.model_m = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )
        self.fc = nn.Linear(6976, input_channels)

    def forward(self, x):
        x = self.model_m(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
