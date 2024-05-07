import torch
import numpy as np
import random
from matplotlib import pyplot as plt
from GAN_model import Generator

# set seed function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return True

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

G = Generator().to(device)
set_seed(0)
z_dim = 100
# load model weight
num = 300
epoch = 600
G.load_state_dict(torch.load("./Generator_norm_{:03d}.pth".format(epoch)))

with torch.no_grad():
    test_z = torch.randn(num, z_dim).to(device)
    generated = G(test_z)

saved_images = generated.view(generated.size(0), 1, 32, 32).cpu()

fig, axarr = plt.subplots(30, 10, figsize=(10, 30))
for ax, img in zip(axarr.flatten(), saved_images):
    ax.axis('off')
    ax.imshow(img.squeeze(), cmap="gray")
plt.title('Epoch = {:03d}'.format(epoch))
plt.show()
