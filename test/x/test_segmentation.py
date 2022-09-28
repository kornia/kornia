from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from kornia.x import Configuration, SemanticSegmentationTrainer


class DummyDatasetSegmentation(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.ones(3, 32, 32), torch.ones(32, 32).long()


@pytest.fixture
def model():
    return nn.Conv2d(3, 10, kernel_size=1)


@pytest.fixture
def dataloader():
    dataset = DummyDatasetSegmentation()
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


class TestsemanticSegmentationTrainer:
    def test_fit(self, model, dataloader, criterion, optimizer, scheduler, configuration):
        trainer = SemanticSegmentationTrainer(
            model, dataloader, dataloader, criterion, optimizer, scheduler, configuration
        )
        trainer.fit()

    def test_exception(self, model, dataloader, criterion, optimizer, scheduler, configuration):
        with pytest.raises(ValueError):
            SemanticSegmentationTrainer(
                model, dataloader, dataloader, criterion, optimizer, scheduler, configuration, callbacks={'frodo': None}
            )
