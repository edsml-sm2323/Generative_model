import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
import tqdm
import time
from vae_model import VAE
from torch.utils.data import DataLoader, Dataset

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(data_loader, model, optimizer, epoch, device):
    train_loss = []
    recon_loss = []
    kld_loss = []
    model.train()
    start_time = time.time()

    for batch_idx,(x, _) in enumerate(tqdm.tqdm(data_loader, desc=f"Training Epoch {epoch}")):

        x = x.to(device)
        print(x.shape)

        optimizer.zero_grad()
        x_tilde, kl_d = model(x)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, x)

        loss = loss_recons + 0.001 * kl_d
        loss.backward()

        optimizer.step()
        train_loss.append(loss.item())
        recon_loss.append(loss_recons.item())
        kld_loss.append(kl_d.item())

    print('Train Completed!\tLoss: {:7.6f}   Reconstruction Loss: {:7.6f}   KLD Loss: {:7.6f}  Time: {:5.3f} s'.format(
        np.asarray(train_loss).mean(0),
        np.asarray(recon_loss).mean(0),
        np.asarray(kld_loss).mean(0),
        time.time() - start_time
    ))


def test(data_loader, model, device):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        loss_recons, loss_kld = 0., 0.
        for batch_idx, (x, _) in enumerate(tqdm.tqdm(data_loader, desc="Validating")):
            x = x.to(device)
            x_tilde, kl_d = model(x)
            loss_recons += F.mse_loss(x_tilde, x)
            loss_kld += kl_d

        loss_recons /= len(data_loader)
        loss_kld /= len(data_loader)

    print('Validation Completed!\tReconstruction Loss: {:7.6f} Time: {:5.3f} s'.format(
        np.array(loss_recons.item()),
        time.time() - start_time
    ))
    return loss_recons.item(), loss_kld.item()


def generate_reconstructions(model, data_loader, device, DATASET):
    model.eval()
    _, x, _ = data_loader.__iter__().next()
    x = x[:4].to(device)
    x_tilde, kl_div = model(x)
    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1) / 2

    save_image(
        images,
        'vae_reconstructions_{}.png'.format(DATASET),
        nrow=4
    )


def main():
    IMAGE_SIZE = 448
    BATCH_SIZE = 64
    N_EPOCHS = 100
    DATASET = 'railway'
    NUM_CHANNEL = 3
    HIDDEN_DIM = 256
    Z_DIM = 4
    LR = 1e-3
    weight_decay = 0

    save_filename = 'data/'

    if not os.path.exists(save_filename):
        os.makedirs(save_filename)

    # load the UWV dataset
    # Get data configuration
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomResizedCrop(448, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define datasets
    train_dataset = ImageFolder(root="Railway Track fault Detection Updated/Train", transform=train_transform)
    val_dataset = ImageFolder(root="Railway Track fault Detection Updated/Validation", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = VAE(NUM_CHANNEL, HIDDEN_DIM, Z_DIM).to(device)
    # load from last time trained file

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    best_loss = -1.
    LAST_SAVED = -1
    for epoch in range(N_EPOCHS):
        train(train_loader, model, optimizer, epoch, device)
        loss, _ = test(valid_loader, model, device)

        # generate_reconstructions(model, valid_loader, device, DATASET)

        if (epoch == 0) or (loss < best_loss):
            print("Saving model!\n")
            best_loss = loss
            LAST_SAVED = epoch
            with open('{0}/{1}_vae.pt'.format(save_filename, DATASET), 'wb') as f:
                torch.save(model.state_dict(), f)
        else:
            print("Not saving model! Last saved: {}\n".format(LAST_SAVED))


if __name__ == '__main__':
    import os
    main()