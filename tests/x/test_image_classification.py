# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from kornia.contrib import ClassificationHead, VisionTransformer
from kornia.x import Configuration, ImageClassifierTrainer
from kornia.x.trainer import Accelerator


class DummyDatasetClassification(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.ones(3, 32, 32), torch.tensor(1)


@pytest.fixture()
def model():
    return nn.Sequential(VisionTransformer(image_size=32), ClassificationHead(num_classes=10))


@pytest.fixture()
def dataloader():
    dataset = DummyDatasetClassification()
    return torch.utils.data.DataLoader(dataset, batch_size=1)


@pytest.fixture()
def criterion():
    return nn.CrossEntropyLoss()


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


class TestImageClassifierTrainer:
    @pytest.mark.slow
    @pytest.mark.skipif(
        torch.__version__ == "1.12.1" and Accelerator is None, reason="accelerate lib problem with torch 1.12.1"
    )
    def test_fit(self, model, dataloader, criterion, optimizer, scheduler, configuration):
        trainer = ImageClassifierTrainer(model, dataloader, dataloader, criterion, optimizer, scheduler, configuration)
        trainer.fit()

    @pytest.mark.skipif(
        torch.__version__ == "1.12.1" and Accelerator is None, reason="accelerate lib problem with torch 1.12.1"
    )
    @pytest.mark.xfail(sys.platform == "darwin", reason="Sometimes CI can fail with MPS backend out of memory")
    def test_exception(self, model, dataloader, criterion, optimizer, scheduler, configuration):
        with pytest.raises(ValueError):
            ImageClassifierTrainer(
                model, dataloader, dataloader, criterion, optimizer, scheduler, configuration, callbacks={"frodo": None}
            )
