"""
Data augmentation on the GPU
============================

In this data you learn how to use `kornia` modules in order to perform the data augmentatio on the GPU in batch mode.
"""

################################
# 1. Create a dummy data loader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from matplotlib import pyplot as plt
import cv2

# from: https://gist.github.com/edgarriba/a781de516c508826f79568d08598efdb


class DummyDataset(Dataset):
    def __init__(self, data_root=None):
        self.data_root = data_root
        self.data_index = self.build_index(self.data_root)

    def build_index(self, data_root):
        return range(10)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        sample = self.data_index[idx]

        # load data, NOTE: modify by cv2.imread(...)
        image = cv2.imread('./data/simba.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return image


def imshow(input: torch.Tensor):
    out: torch.Tensor = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = kornia.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()

################################
# 2. Define the data augmentation operations
# Thanks to the `kornia` design all the operators can be placed inside inside a `nn.Sequential`.

import kornia

transform = kornia.augmentation.augmentations.ColorJitter(brightness=0.5, contrast=0.5, hue=0.05, saturation=.5)
print(transform)

################################
# 3. Run the dataset and perform the data augmentation

# NOTE: change device to 'cuda'
device = torch.device('cpu')
print(f"Running with device: {device}")

# create the dataloader
dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# get samples and perform the data augmentation
for i_batch, images in enumerate(dataloader):
    images = images.to(device) / 255.0

    # perform the transforms
    images_out = transform(images)

    print(f"Iteration: {i_batch} Image shape: {images.shape}")
    for j in range(images.shape[0]):
        imshow(images[j])
        imshow(images_out[j])
