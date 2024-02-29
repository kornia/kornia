import pytest
from torch import nn

from kornia.metrics import AverageMeter
from kornia.x import EarlyStopping, ModelCheckpoint
from kornia.x.utils import TrainerState


@pytest.fixture()
def model():
    return nn.Conv2d(3, 10, kernel_size=1)


class TestModelCheckpoint:
    def test_smoke(self, tmp_path, model):
        cb = ModelCheckpoint(tmp_path, "test_monitor", reverse_metric=True)
        assert cb is not None

        metric = {"test_monitor": AverageMeter()}
        metric["test_monitor"]._avg = 1.0

        cb(model, epoch=0, valid_metric=metric)
        assert cb.best_metric == 1.0
        assert (tmp_path / "model_epoch=0_loss=1.0.pt").is_file()

    def test_custom_filename(self, tmp_path, model):
        cb = ModelCheckpoint(tmp_path, "test_monitor", filename_fcn=lambda x: "model.pt", reverse_metric=True)
        assert cb is not None

        metric = {"test_monitor": AverageMeter()}
        metric["test_monitor"]._avg = 1.0

        cb(model, epoch=0, valid_metric=metric)
        assert cb.best_metric == 1.0
        assert (tmp_path / "model.pt").is_file()


def test_callback_earlystopping(model):
    cb = EarlyStopping("test_monitor", patience=2, reverse_metric=True)
    assert cb is not None
    assert cb.counter == 0

    metric = {"test_monitor": AverageMeter()}
    metric["test_monitor"]._avg = 1

    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TRAINING
    assert cb.best_score == 1
    assert cb.counter == 0

    metric["test_monitor"]._avg = 2
    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TRAINING
    assert cb.best_score == 2
    assert cb.counter == 0

    metric["test_monitor"]._avg = 1.9
    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TRAINING
    assert cb.best_score == 2
    assert cb.counter == 1

    metric["test_monitor"]._avg = 1.9
    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TERMINATE
    assert cb.best_score == 2
    assert cb.counter == 2

def test_callback_earlystopping(model):
    cb = EarlyStopping("test_monitor", patience=2, reverse_metric=False)
    assert cb is not None
    assert cb.counter == 0

    metric = {"test_monitor": AverageMeter()}
    metric["test_monitor"]._avg = -1

    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TRAINING
    assert cb.best_score == 1
    assert cb.counter == 0

    metric["test_monitor"]._avg = -2
    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TRAINING
    assert cb.best_score == 2
    assert cb.counter == 0

    metric["test_monitor"]._avg = -1.9
    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TRAINING
    assert cb.best_score == 2
    assert cb.counter == 1

    metric["test_monitor"]._avg = -1.9
    state = cb(model, epoch=0, valid_metric=metric)
    assert state == TrainerState.TERMINATE
    assert cb.best_score == 2
    assert cb.counter == 2