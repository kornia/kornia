"""
Data augmentation on the GPU
============================

In this data you learn how to use `kornia` modules in order to perform the data augmentation on the GPU in batch mode.
"""

################################
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import Resize
import kornia
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob


################################
# 1. Create dataset
class DummyDataset():
    def __init__(self, root, transform=None):

        self.images = sorted(glob.glob(root + "/*.jpg"))
        self.targets = sorted(glob.glob(root + "/*.pth"))
        self.transforms = transform

    def __len__(self):
        return 4

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert("RGB")
        target = torch.load(self.targets[index])

        ow, oh = img.size

        # Reescale boxes
        target[:2].mul_(512 / ow)
        target[-2:].mul_(512 / oh)

        # Reorder boxes as xyxy
        perm = torch.LongTensor([0, 2, 1, 3])
        target = target[perm]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


# Utils functions
def plot_images(org, img):

    fig, ax = plt.subplots(2, img.shape[0], sharex=True, sharey=True)
    fig.suptitle("Original (Top) / Trasnformed (Bottom)", fontsize=16)
    for b in range(img.shape[0]):
        ax[0, b].imshow(org[b])
        ax[1, b].imshow(img[b])

    plt.xticks([])
    plt.yticks([])
    plt.show()


def draw_boxes(original, images, transformed_targets, targets):

    for i in range(original.shape[0]):
        images[i] = cv2.rectangle(
            images[i],
            (transformed_targets[i][0], transformed_targets[i][1]),
            (transformed_targets[i][2], transformed_targets[i][3]),
            (0, 1, 0),
            2,
        ).get()
        original[i] = cv2.rectangle(
            original[i],
            (targets[i][0], targets[i][1]),
            (targets[i][2], targets[i][3]),
            (0, 1, 0),
            2,
        ).get()

    return original, images


def transform_boxes(mat, targets):

    return kornia.transform_points(mat, targets.view(-1, 2, 2).float()).view(-1, 4)


################################
# 2. Define the data augmentation operations
# Thanks to the `kornia` design all the operators can be placed inside inside a `nn.Sequential`.

transform = nn.Sequential(
    kornia.augmentation.ColorJitter(
        brightness=(0.0, 0.0),
        contrast=(1.0, 1.0),
        hue=1.5,
        saturation=2.0,
        return_transform=True,
    ),
    kornia.augmentation.RandomHorizontalFlip(1.0, return_transform=True),
)

################################
# 3. Run the dataset and perform the data augmentation

device = torch.device("cuda")
seed = 0
torch.manual_seed(seed)
print(f"Running with device: {device}")
print(f"Running with seed: {seed}")

# create the dataloader
batch_size = 4
dataset = DummyDataset(
    root="data/aug",
    transform=lambda x: kornia.image_to_tensor(Resize((512, 512))(x)),
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# get next batch
images, targets = next(iter(dataloader))

# move tensors to GPU
images = images.to(device) / 255
targets = targets.to(device)

# keep a copy of the original image
original = kornia.tensor_to_image(images)

# perform the transforms
images, mat = transform(images)
images = kornia.tensor_to_image(images)

plot_images(original, images)

# transform boxes
targets[:, 2] = targets[:, 2] - targets[:, 0]
targets[:, 3] = targets[:, 3] - targets[:, 1]

transformed_targets = kornia.transform_boxes(mat, targets, mode="xywh")

transformed_targets[..., -2] = transformed_targets[..., 0] + transformed_targets[..., -2]  # x + w
transformed_targets[..., -1] = transformed_targets[..., 1] + transformed_targets[..., -1]  # y + h

# draw boxes
original, images = draw_boxes(original, images, transformed_targets, targets)
plot_images(original, images)
