import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from kornia.x import Configuration, ObjectDetectionTrainer
from kornia.x.trainer import Accelerator


class DummyDatasetDetection(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.ones(3, 32, 32), torch.tensor([10.0, 10.0, 15.0, 15.0])


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 1, kernel_size=16, stride=16), nn.Flatten())

    def forward(self, x, y=None):
        return self.model(x)


@pytest.fixture()
def model():
    return DummyModel()


@pytest.fixture()
def dataloader():
    dataset = DummyDatasetDetection()
    return torch.utils.data.DataLoader(dataset, batch_size=1)


@pytest.fixture()
def criterion():
    return nn.MSELoss()


@pytest.fixture()
def optimizer(model):
    return torch.optim.AdamW(model.parameters())


@pytest.fixture()
def scheduler(optimizer, dataloader):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))


@pytest.fixture()
def configuration():
    config = Configuration()
    config.num_epochs = 1
    return config


class TestObjectDetectionTrainer:
    @pytest.mark.slow
    @pytest.mark.skipif(
        torch.__version__ == '1.12.1' and Accelerator is None, reason='accelerate lib problem with torch 1.12.1'
    )
    @pytest.mark.parametrize("loss_computed_by_model", [True, False])
    def test_fit(self, model, dataloader, criterion, optimizer, scheduler, configuration, loss_computed_by_model):
        trainer = ObjectDetectionTrainer(
            model,
            dataloader,
            dataloader,
            criterion,
            optimizer,
            scheduler,
            configuration,
            num_classes=3,
            loss_computed_by_model=loss_computed_by_model,
        )
        trainer.fit()

    @pytest.mark.skipif(
        torch.__version__ == '1.12.1' and Accelerator is None, reason='accelerate lib problem with torch 1.12.1'
    )
    def test_exception(self, model, dataloader, criterion, optimizer, scheduler, configuration):
        with pytest.raises(ValueError):
            ObjectDetectionTrainer(
                model,
                dataloader,
                dataloader,
                criterion,
                optimizer,
                scheduler,
                configuration,
                num_classes=3,
                callbacks={'frodo': None},
            )
