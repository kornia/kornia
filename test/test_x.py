import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from kornia.contrib import ClassificationHead, VisionTransformer
from kornia.metrics import AverageMeter
from kornia.x import Configuration, EarlyStopping, ImageClassifierTrainer, ModelCheckpoint
from kornia.x.utils import TrainerState


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


@pytest.fixture
def dataloader(dataset):
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
        trainer = ImageClassifierTrainer(
            model, dataloader, dataloader, criterion, optimizer, scheduler, configuration,
        )
        trainer.fit()

    def test_exception(self, model, dataloader, criterion, optimizer, scheduler, configuration):
        with pytest.raises(ValueError):
            ImageClassifierTrainer(
                model, dataloader, dataloader, criterion, optimizer, scheduler, configuration,
                callbacks={'frodo': None},
            )


def test_callback_modelcheckpoint(tmp_path, model):
    cb = ModelCheckpoint(tmp_path, 'test_monitor')
    assert cb is not None

    metric = {'test_monitor': AverageMeter()}
    metric['test_monitor'].avg = 1.

    cb(model, epoch=0, valid_metric=metric)
    assert cb.best_metric == 1.0
    assert (tmp_path / "model_0.pt").is_file()


def test_callback_earlystopping(model):
    cb = EarlyStopping('test_monitor', patience=2)
    assert cb is not None
    assert cb.counter == 0

    metric = {'test_monitor': AverageMeter()}
    metric['test_monitor'].avg = 1

    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TRAINING
    assert cb.best_score == -1
    assert cb.counter == 0

    metric['test_monitor'].avg = 2
    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TRAINING
    assert cb.best_score == -1
    assert cb.counter == 1

    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TERMINATE
