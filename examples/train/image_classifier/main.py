from __future__ import annotations

import hydra
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

import kornia as K
from kornia.x import Configuration, ImageClassifierTrainer, ModelCheckpoint

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Configuration)


@hydra.main(config_path=".", config_name="config.yaml")
def my_app(config: Configuration) -> None:
    # create the model
    model = nn.Sequential(
        K.contrib.VisionTransformer(image_size=32, patch_size=16, embed_dim=128, num_heads=3),
        K.contrib.ClassificationHead(embed_size=128, num_classes=10),
    )

    # create the dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=to_absolute_path(config.data_path), train=True, download=True, transform=T.ToTensor()
    )

    valid_dataset = torchvision.datasets.CIFAR10(
        root=to_absolute_path(config.data_path), train=False, download=True, transform=T.ToTensor()
    )

    # create the dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    valid_daloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    # create the loss function
    criterion = nn.CrossEntropyLoss()

    # instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs * len(train_dataloader))

    # define some augmentations
    _augmentations = nn.Sequential(
        K.augmentation.RandomHorizontalFlip(p=0.75),
        K.augmentation.RandomVerticalFlip(p=0.75),
        K.augmentation.RandomAffine(degrees=10.0),
        K.augmentation.PatchSequential(
            K.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.8),
            grid_size=(2, 2),  # cifar-10 is 32x32 and vit is patch 16
            patchwise_apply=False,
        ),
    )

    def augmentations(self, sample: dict) -> dict:
        out = _augmentations(sample["input"])
        return {"input": out, "target": sample["target"]}

    model_checkpoint = ModelCheckpoint(filepath="./outputs", monitor="top5")

    trainer = ImageClassifierTrainer(
        model,
        train_dataloader,
        valid_daloader,
        criterion,
        optimizer,
        scheduler,
        config,
        callbacks={"augmentations": augmentations, "on_checkpoint": model_checkpoint},
    )
    trainer.fit()


if __name__ == "__main__":
    my_app()
