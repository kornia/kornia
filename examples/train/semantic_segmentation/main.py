import hydra
import numpy as np
import torch
import torch.nn as nn
import torchvision
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

import kornia as K
from kornia.x import Configuration, Lambda, SemanticSegmentationTrainer

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Configuration)


class Transform(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.resize = K.geometry.Resize(image_size, interpolation='nearest')

    @torch.no_grad()
    def forward(self, x, y):
        x = K.utils.image_to_tensor(np.array(x))
        x, y = x.float() / 255.0, torch.from_numpy(y)
        return self.resize(x), self.resize(y)


@hydra.main(config_path=".", config_name="config.yaml")
def my_app(config: Configuration) -> None:

    # make image size homogeneous
    transform = Transform(tuple(config.image_size))

    # create the dataset
    train_dataset = torchvision.datasets.SBDataset(
        root=to_absolute_path(config.data_path), image_set='train', download=False, transforms=transform
    )

    valid_dataset = torchvision.datasets.SBDataset(
        root=to_absolute_path(config.data_path), image_set='val', download=False, transforms=transform
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

    # create the model
    model = nn.Sequential(torchvision.models.segmentation.fcn_resnet50(pretrained=False), Lambda(lambda x: x['out']))

    # instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs * len(train_dataloader))

    # define some augmentations
    _augmentations = K.augmentation.AugmentationSequential(
        K.augmentation.RandomHorizontalFlip(p=0.75),
        K.augmentation.RandomVerticalFlip(p=0.75),
        K.augmentation.RandomAffine(degrees=10.0),
        data_keys=['input', 'mask'],
    )

    def preprocess(self, sample: dict) -> dict:
        target = sample["target"].argmax(1).unsqueeze(1).float()
        return {"input": sample["input"], "target": target}

    def augmentations(self, sample: dict) -> dict:
        x, y = _augmentations(sample["input"], sample["target"])
        # NOTE: use matplotlib to visualise before/after samples
        return {"input": x, "target": y}

    def on_before_model(self, sample: dict) -> dict:
        target = sample["target"].squeeze(1).long()
        return {"input": sample["input"], "target": target}

    trainer = SemanticSegmentationTrainer(
        model,
        train_dataloader,
        valid_daloader,
        criterion,
        optimizer,
        scheduler,
        config,
        callbacks={"preprocess": preprocess, "augmentations": augmentations, "on_before_model": on_before_model},
    )
    trainer.fit()


if __name__ == "__main__":
    my_app()
