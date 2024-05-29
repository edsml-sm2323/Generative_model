import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["CustomDataset"]


class CustomDataset(Dataset):
    """
    Custom dataset for loading sequences of images and corresponding targets from .npy files.

    Parameters:
    - data_file (str): Path to the .npy file containing image data.
    - sequence_length (int): Number of images in each input sequence (default is 4).
    - step (int): Step size for selecting images in each sequence (default is 4).
    - transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data_file, sequence_length=4, step=4, transform=None):
        self.data = np.load(data_file)
        self.sequence_length = sequence_length
        self.step = step
        self.transform = transform
        self.data_len = len(self.data) - (self.sequence_length * self.step)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        input_indices = [idx + i * self.step for i in range(self.sequence_length)]
        target_index = idx + self.sequence_length * self.step

        input_images = self.data[input_indices]
        target_image = self.data[target_index]

        if self.transform:
            input_images = [self.transform(image) for image in input_images]
            target_image = self.transform(target_image)

        # Convert to PyTorch tensors
        input_images = torch.tensor(input_images, dtype=torch.float32).unsqueeze(1)
        target_image = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0)

        return input_images, target_image
