import hydra
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

import kornia as K
from kornia.x import Configuration, EarlyStopping, SemanticSegmentationTrainer, ModelCheckpoint

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Configuration)


@hydra.main(config_path=".", config_name="config.yaml")
def my_app(config: Configuration) -> None:

    # create the model
    model = nn.Sequential(
        K.contrib.VisionTransformer(image_size=224, patch_size=16),
        #K.contrib.ClassificationHead(embed_size=128, num_classes=10),
    )

    def tf(x, y):
        return T.ToTensor()(x)[..., :224, :224], torch.from_numpy(y).float()[..., :224, :224]

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
    augmentations = K.augmentation.AugmentationSequential(
        K.augmentation.RandomHorizontalFlip(p=0.75),
        K.augmentation.RandomVerticalFlip(p=0.75),
        # K.augmentation.RandomAffine(degrees=10.),
        data_keys=['input', 'mask']
    )

    model_checkpoint = ModelCheckpoint(
        filepath="./outputs", monitor="top5",
    )

    early_stop = EarlyStopping(monitor="top5")

    trainer = SemanticSegmentationTrainer(
        model, train_dataloader, valid_daloader, criterion, optimizer, scheduler, config,
        callbacks={
            "augmentations": augmentations,
            "checkpoint": model_checkpoint,  # "terminate": early_stop,
        }
    )
    trainer.fit()

if __name__ == "__main__":
    my_app()
