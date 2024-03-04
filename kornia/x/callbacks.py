from math import inf
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch

from kornia.core import Module
from kornia.metrics import AverageMeter

from .utils import TrainerState


# default function to generate the filename in the model checkpoint
def default_filename_fcn(epoch: Union[str, int], metric: Union[str, float]) -> str:
    return f"model_epoch={epoch}_metricValue={metric}.pt"


class EarlyStopping:
    """Callback that evaluates whether there is improvement in the loss function.

    The module track the losses and in case of finish patience sends a termination signal to the trainer.

    Args:
        monitor: the name of the value to track.
        min_delta: the minimum difference between losses to increase the patience counter.
        patience: the number of times to wait until the trainer does not terminate.
        max_mode: if true metric will be multiply by -1,
                  turn this flag when increasing metric value is expected for example Accuracy

    **Usage example:**

    .. code:: python

        early_stop = EarlyStopping(
            monitor="loss", patience=10
        )

        trainer = ImageClassifierTrainer(
            callbacks={"on_epoch_end", early_stop}
        )
    """

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 8,
        max_mode: bool = False,
    ) -> None:
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        # flag to reverse metric, for example in case of accuracy metric where bigger value is better
        # In classical loss functions smaller value = better,
        # in case of max_mode training end with metric stable/decreasing
        self.max_mode = max_mode

        self.counter: int = 0
        self.best_score: float = -inf if max_mode else inf
        self.early_stop: bool = False

    def __call__(self, model: Module, epoch: int, valid_metric: Dict[str, AverageMeter]) -> TrainerState:
        score: float = valid_metric[self.monitor].avg
        is_best: bool = score > self.best_score if self.max_mode else score < self.best_score
        if is_best:
            self.best_score = score
            self.counter = 0
        else:
            # Example score = 1.9 best_score = 2.0 min_delta = 0.15
            # with max_mode (1.9 > (2.0 - 0.15)) == True
            # with min_mode (1.9 < (2.0 + 0.15)) == True
            is_within_delta: bool = (
                score > (self.best_score - self.min_delta)
                if self.max_mode
                else score < (self.best_score + self.min_delta)
            )
            if not is_within_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        if self.early_stop:
            print(f"[INFO] Early-Stopping the training process. Epoch: {epoch}.")
            return TrainerState.TERMINATE

        return TrainerState.TRAINING


class ModelCheckpoint:
    """Callback that save the model at the end of every epoch.

    Args:
        filepath: the where to save the mode.
        monitor: the name of the value to track.
        max_mode: if true metric will be multiply by -1
                  turn this flag when increasing metric value is expected for example Accuracy
    **Usage example:**

    .. code:: python

        model_checkpoint = ModelCheckpoint(
            filepath="./outputs", monitor="loss",
        )

        trainer = ImageClassifierTrainer(...,
            callbacks={"on_checkpoint", model_checkpoint}
        )
    """

    def __init__(
        self,
        filepath: str,
        monitor: str,
        filename_fcn: Optional[Callable[..., str]] = None,
        max_mode: bool = False,
    ) -> None:
        self.filepath = filepath
        self.monitor = monitor
        self._filename_fcn = filename_fcn or default_filename_fcn
        # track best model
        self.best_metric: float = -inf if max_mode else inf
        # flag to reverse metric, for example in case of accuracy metric where bigger value is better
        # In classical loss functions smaller value = better,
        # In case of max_mode checkpoints are saved if new metric value > old metric value
        self.max_mode = max_mode

        # create directory
        Path(self.filepath).mkdir(parents=True, exist_ok=True)

    def __call__(self, model: Module, epoch: int, valid_metric: Dict[str, AverageMeter]) -> None:
        valid_metric_value: float = valid_metric[self.monitor].avg
        is_best: bool = (
            valid_metric_value > self.best_metric if self.max_mode else valid_metric_value < self.best_metric
        )
        if is_best:
            self.best_metric = valid_metric_value
            # store old metric and save new model
            filename = Path(self.filepath) / self._filename_fcn(epoch, valid_metric_value)
            torch.save(model, filename)
