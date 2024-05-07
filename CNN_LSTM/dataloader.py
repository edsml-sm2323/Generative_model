import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import os

__all__ = ['CustomDataset', 'train_test_split', 'create_combined_dataset']


class CustomDataset(Dataset):

    """
        A custom dataset class for
        loading and transforming a sequence of images.

        Attributes:
        image_folder (str): Directory containing images.
        sequence (int): Number of consecutive images to be considered as input.
        transform (callable, optional):
        Optional transform to be applied on a sample.

        Methods:
        __len__: Returns the total number of samples in the dataset.
        __getitem__: Retrieves a sample with a given index.
        """

    def __init__(self, image_folder, squence, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_names = sorted([file for file in os.listdir(image_folder) if file.endswith('.jpg')]) # noqa
        self.squence = squence
        self.target_sequence = 3

    def __len__(self):
        return len(self.image_names) - self.squence - self.target_sequence

    def __getitem__(self, idx):
        sequence_files = self.image_names[idx:idx + self.squence + self.target_sequence] # noqa
        images = [Image.open(os.path.join(self.image_folder, file)) for file in sequence_files] # noqa
        if self.transform is not None:
            images = [self.transform(image) for image in images]

        input_images = images[:self.squence]
        target_image = images[self.squence:]

        # Convert to PyTorch tensors
        input_images = torch.stack(input_images)
        target_image = torch.stack(target_image)

        return input_images, target_image


def train_test_split(dataset, test_ratio=0.2):
    """
        Splits a dataset into training and testing sets.

        Parameters:
        dataset (Dataset): The dataset to split.
        test_ratio (float, optional):
        The proportion of the dataset to include in the test split.

        Returns:
        tuple: Tuple containing the training set and the test set.
        """

    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size

    return torch.utils.data.random_split(dataset, [train_size, test_size])


def create_combined_dataset(parent_folder_path, sequence, transform=None):

    """
        Creates a combined dataset from
        multiple subfolders within a parent folder.

        Parameters:
        parent_folder_path (str):
        Path to the parent folder containing subfolders with images.
        sequence (int):
        Number of consecutive images to be considered as input.
        transform (callable, optional):
        Optional transform to be applied on a sample.

        Returns:
        ConcatDataset:
        A concatenated dataset consisting of
        datasets from all subfolders.
        """

    all_datasets = []

    for subfolder_name in os.listdir(parent_folder_path):
        subfolder_path = os.path.join(parent_folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            current_dataset = CustomDataset(subfolder_path,
                                            sequence,
                                            transform=transform)
            all_datasets.append(current_dataset)

    combined_dataset = ConcatDataset(all_datasets)

    return combined_dataset