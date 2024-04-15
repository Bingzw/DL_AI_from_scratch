import torch
import torch.nn as nn
import pytorch_lightning as pl


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CNN(nn.Module):
    def __init__(self, hidden_features=32, out_dim=1, **kwargs):
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=1, padding=4),  # (28 + 2*padding - kernel_size - 1 - 1)/stride + 1 = 16
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),  # 8*8
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),  # 4*4
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=4, stride=1, padding=0),  # 2*2
            Swish(),
            nn.Flatten(),
            nn.Linear(c_hid3*4, c_hid3),
            Swish(),
            nn.Linear(c_hid3, out_dim)
        )

    def forward(self, x):
        return self.cnn_layers(x).squeeze(dim=-1)


class DeepEnergyModel(pl.LightningModule):
    def __init__(self):
        pass
