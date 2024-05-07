import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from vae_model import VAE
from torchvision.transforms import transforms


weights_path = 'railway_vae_big.pt'
DATASET = 'coco128'
device = 'cpu'
IMAGE_SIZE = 448
NUM_CHANNEL = 3
HIDDEN_DIM = 256
Z_DIM = 4


test_transform = transforms.Compose([
    transforms.Resize(448),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = VAE(NUM_CHANNEL, HIDDEN_DIM, Z_DIM).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
# Define the test dataset
test_dataset = ImageFolder(root="Railway Track fault Detection Updated/Test", transform=test_transform)

# Define the data loader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


def generate_reconstructions(model, data_loader, device, DATASET):
    model.eval()
    x, _ = next(iter(data_loader))
    x = x[:4].to(device)
    x_tilde, kl_div = model(x)
    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1)/2

    save_image(
        images,
        'vae_reconstructions_{}.png'.format(DATASET),
        nrow=4
    )


generate_reconstructions(model, test_loader,device,DATASET)