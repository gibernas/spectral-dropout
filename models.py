import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import to_spatial, to_spectral


class VanillaCNN(nn.Module):
    def __init__(self, as_gray=True,):
        super(VanillaCNN, self).__init__()

        # Name of the model
        self.name = 'VanillaCNN'

        # Handle dimensions
        if as_gray:
            self.input_channels = 1
        else:
            self.input_channels = 3

        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64))

        self.drop_out_lin1 = nn.Dropout(0.4)
        self.lin1 = nn.Linear(1152, 512)
        self.drop_out_lin2 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(512, 256)
        self.drop_out_lin3 = nn.Dropout(0.1)
        self.lin3 = nn.Linear(256, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out_lin1(out)
        out = F.relu(self.lin1(out))
        out = self.drop_out_lin2(out)
        out = F.relu(self.lin2(out))
        out = self.drop_out_lin3(out)
        out = torch.tanh(self.lin3(out))

        return out


class SpectralDropoutEasyCNN(nn.Module):
    def __init__(self, as_gray=True, dev='cpu'):
        super(SpectralDropoutEasyCNN, self).__init__()

        # Name of the model
        self.name = 'SpectralDropoutCNN'
        self.dev = dev

        # Handle dimensions
        if as_gray:
            self.input_channels = 1
        else:
            self.input_channels = 3

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64))

        self.drop_out_lin1 = nn.Dropout(0.4)
        self.lin1 = nn.Linear(1152, 512)
        self.drop_out_lin2 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(512, 256)
        self.drop_out_lin3 = nn.Dropout(0.1)
        self.lin3 = nn.Linear(256, 2)
        self.dropout_layer = nn.Dropout2d(p=0.2)        

    def forward(self, x):
        out = to_spectral(x.cpu())
        out = self.dropout_layer(out)
        out = to_spatial(out.cpu())
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out_lin1(out)
        out = F.relu(self.lin1(out))
        out = self.drop_out_lin2(out)
        out = F.relu(self.lin2(out))
        out = self.drop_out_lin3(out)
        out = torch.tanh(self.lin3(out))

        return out

