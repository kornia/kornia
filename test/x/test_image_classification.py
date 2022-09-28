from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from kornia.contrib import ClassificationHead, VisionTransformer
from kornia.x import Configuration, ImageClassifierTrainer


class DummyDatasetClassification(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.ones(3, 32, 32), torch.tensor(1)


@pytest.fixture
def model():
    return nn.Sequential(VisionTransformer(image_size=32), ClassificationHead(num_classes=10))


@pytest.fixture
def dataloader():
    dataset = DummyDatasetClassification()
    return torch.utils.data.DataLoader(dataset, batch_size=1)


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(model):
    return torch.optim.AdamW(model.parameters())


@pytest.fixture
def scheduler(optimizer, dataloader):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))


@pytest.fixture
def configuration():
    config = Configuration()
    config.num_epochs = 1
    return config


class TestImageClassifierTrainer:
    def test_fit(self, model, dataloader, criterion, optimizer, scheduler, configuration):
        trainer = ImageClassifierTrainer(model, dataloader, dataloader, criterion, optimizer, scheduler, configuration)
        trainer.fit()

    def test_exception(self, model, dataloader, criterion, optimizer, scheduler, configuration):
        with pytest.raises(ValueError):
            ImageClassifierTrainer(
                model, dataloader, dataloader, criterion, optimizer, scheduler, configuration, callbacks={'frodo': None}
            )
