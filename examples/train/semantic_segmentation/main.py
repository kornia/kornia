from kornia import augmentation
import hydra
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

import kornia as K
from kornia.x import (
    Configuration, SemanticSegmentationTrainer, ModelCheckpoint, Lambda
)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Configuration)


@hydra.main(config_path=".", config_name="config.yaml")
def my_app(config: Configuration) -> None:

    # create the model
    model = nn.Sequential(
        torchvision.models.segmentation.fcn_resnet50(pretrained=False),
        Lambda(lambda x: x['out']),
    )

    def tf(x, y) -> dict:
        return T.ToTensor()(x), torch.from_numpy(y)

    # create the dataset
    train_dataset = torchvision.datasets.SBDataset(
        root=to_absolute_path(config.data_path), image_set='train', download=False, transforms=tf)

    valid_dataset = torchvision.datasets.SBDataset(
        root=to_absolute_path(config.data_path), image_set='val', download=False, transforms=tf)

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
    _augmentations = K.augmentation.AugmentationSequential(
        K.augmentation.RandomHorizontalFlip(p=0.75),
        K.augmentation.RandomVerticalFlip(p=0.75),
        K.augmentation.RandomAffine(degrees=10.),
        data_keys=['input', 'mask']
    )

    def preprocess(sample: dict) -> dict:
        target = sample["target"].argmax(1).unsqueeze(0).float()
        return {"input": sample["input"], "target": target}

    def augmentations(sample: dict) -> dict:
        x, y = _augmentations(sample["input"], sample["target"])
        return {"input": x, "target": y}

    def postprocess(sample: dict) -> dict:
        target = sample["target"].squeeze(1).long()
        return {"input": sample["input"], "target": target}

    model_checkpoint = ModelCheckpoint(
        filepath="./outputs", monitor="iou",
    )

    trainer = SemanticSegmentationTrainer(
        model, train_dataloader, valid_daloader, criterion, optimizer, scheduler, config,
        callbacks={
            "preprocess": preprocess,
            "augmentations": augmentations,
            "postprocess": postprocess,
            "checkpoint": model_checkpoint,
        }
    )
    trainer.fit()

if __name__ == "__main__":
    my_app()