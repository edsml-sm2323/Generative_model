import torch.nn as nn
import torch
import os
from PIL import Image
import numpy as np
from GAN_model import Generator, Discriminator
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms


# data processing
class Mydataset(TensorDataset):
    def __init__(self, root, transform=None):
        # Determine file path
        self.root = root
        # toTensor and Normaliszation
        self.transform = transform
        # get image from the folder
        self.images = [f for f in os.listdir(root) if f.endswith(('.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.images[idx])
        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        return image


# get the dataloader
transform_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
bs = 32
rs = Mydataset(root='real_hands', transform=transform_2)
train_loader = DataLoader(dataset=rs, batch_size=bs, shuffle=True)


# set the device
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


# define the model
G = Generator().to(device)
D = Discriminator().to(device)
z_dim = 100


# loss function
criterion = nn.BCELoss()


# optimizer
glr = 0.0001
dlr = 0.00009
G_optimizer = torch.optim.Adam(G.parameters(), lr = glr)
D_optimizer = torch.optim.Adam(D.parameters(), lr = dlr)


# Discriminator training
def D_train(x):
    D.train()
    D_optimizer.zero_grad()
    x_real, y_real = x.view(-1, 32*32), torch.ones(bs, 1)
    x_real, y_real = x_real.to(device), y_real.to(device)
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    z = torch.randn(bs, z_dim).to(device)
    x_fake, y_fake = G(z), torch.zeros(bs, 1).to(device)
    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    return  D_loss.data.item()


# Generator training
def G_train(x):
    G.train()
    G_optimizer.zero_grad()
    current_bs = x.size(0)

    z = torch.randn(current_bs, z_dim).to(device)
    G_output = G(z).to(device)

    D_output = D(G_output).to(device)

    y = torch.ones(current_bs, 1).to(device)
    G_loss = criterion(D_output, y).to(device)

    G_loss.backward()
    G_optimizer.step()
    return G_loss.data.item()


# training process
n_epoch =30
groups = {'Loss': ['D_Loss', 'G_Loss']}

for epoch in range(1, n_epoch+1):
  D_losses, G_losses = [], []
  logs = {}
  for batch_idx, x in enumerate(train_loader):
    logs['D_Loss'] = D_train(x)
    logs['G_Loss'] = G_train(x)
    print(batch_idx)
  if(np.mod(epoch, 20) == 0):
    torch.save(G.state_dict(), "./GeneratordcDAN1_{:03d}.pth".format(epoch))