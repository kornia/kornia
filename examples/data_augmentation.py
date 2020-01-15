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
        image = torch.rand(3, 240, 320)
        label = torch.rand(1, 240, 320)
        return dict(images=image, labels=label)

################################
# 2. Define the data augmentation operations
# Thanks to the `kornia` design all the operators can be placed inside inside a `nn.Sequential`.

import kornia

transform = nn.Sequential(
    kornia.color.AdjustBrightness(0.5),
    kornia.color.AdjustGamma(gamma=2.),
    kornia.color.AdjustContrast(0.7),
)

################################
# 3. Run the dataset and perform the data augmentation

# NOTE: change device to 'cuda'
device = torch.device('cpu')
print(f"Running with device: {device}")

# create the dataloader
dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# get samples and perform the data augmentation
for i_batch, sample_batched in enumerate(dataloader):
    images = sample_batched['images'].to(device)
    labels = sample_batched['labels'].to(device)

    # perform the transforms
    images = transform(images)
    labels = transform(labels)

    print(f"Iteration: {i_batch} Image shape: {images.shape}")
