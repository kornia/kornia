from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import kornia as K
from kornia.x import Configuration, EarlyStopping, ImageClassifierTrainer, ModelCheckpoint

# Experiment hyperparameters
config = Configuration()
config.data_path = Path(__file__).absolute().parent
config.batch_size = 64
config.num_epochs = 250
config.lr = 1e-3
config.output_path = "./output"

# create the model
model = nn.Sequential(
    K.contrib.VisionTransformer(image_size=32, patch_size=16, embed_dim=128, num_heads=3),
    K.contrib.ClassificationHead(embed_size=128, num_classes=10),
)

# create the dataset
train_dataset = torchvision.datasets.CIFAR10(
    root=config.data_path, train=True, download=True, transform=T.ToTensor())

valid_dataset = torchvision.datasets.CIFAR10(
    root=config.data_path, train=False, download=True, transform=T.ToTensor())

# create the dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)

valid_daloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)

# create the loss function
criterion = nn.CrossEntropyLoss()

# instantiate the optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, config.num_epochs * len(train_dataloader))

# define some augmentations
augmentations = nn.Sequential(
    K.augmentation.RandomHorizontalFlip(p=0.75),
    K.augmentation.RandomVerticalFlip(p=0.75),
    K.augmentation.RandomAffine(degrees=10.),
    K.augmentation.PatchSequential(
        K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.8),
        grid_size=(2, 2),  # cifar-10 is 32x32 and vit is patch 16
        patchwise_apply=False,
    ),
)

model_checkpoint = ModelCheckpoint(
    filepath="./outputs", monitor="top5",
)

early_stop = EarlyStopping(
    monitor="top5", filepath="early_stop_model.pt"
)

trainer = ImageClassifierTrainer(
    model, train_dataloader, valid_daloader, criterion, optimizer, scheduler, config,
    callbacks={
        "augmentations": augmentations,
        # "checkpoint": model_checkpoint, "terminate": early_stop,
    }
)
trainer.fit()
