import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_input_dim=100, g_output_dim=32 * 32):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 128 * 8 * 8)
        self.conv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 128, 8, 8)
        x = F.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = x.view(-1, 32 * 32)
        return x


class Discriminator(nn.Module):
    def __init__(self, d_input_dim=32 * 32):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(-1, 128 * 8 * 8)
        x = torch.sigmoid(self.fc1(x))
        return x
