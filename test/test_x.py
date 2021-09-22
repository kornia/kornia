import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from kornia.contrib import ClassificationHead, VisionTransformer
from kornia.x import Configuration, ImageClassifierTrainer


class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.ones(3, 32, 32), torch.tensor(1)


@pytest.fixture
def model():
    return nn.Sequential(
        VisionTransformer(image_size=32),
        ClassificationHead(num_classes=10),
    )


@pytest.fixture
def dataset():
    return DummyDataset()


def test_image_classifier(model, dataset):

    config = Configuration()
    config.num_epochs = 1

    # create the dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # create the loss function
    criterion = nn.CrossEntropyLoss()

    # instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.num_epochs * len(train_dataloader))

    trainer = ImageClassifierTrainer(
        model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, config,
    )
    trainer.fit()
