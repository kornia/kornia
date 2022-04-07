from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .utils import TrainerState


class EarlyStopping:
    """Callback that evaluates whether there is improvement in the loss function.

    The module track the losses and in case of finish patience sends a termination signal to the trainer.
    In case of termination, the module will save the last model.

    Args:
        monitor: the name of the value to track.
        min_delta: the minimum difference between losses to increase the patience counter.
        patience: the number of times to wait until the trainer does not terminate.
        filepath: a backup filename to save the file in case of termination.

    **Usage example:**

    .. code:: python

        early_stop = EarlyStopping(
            monitor="top5", filepath="early_stop_model.pt"
        )

        trainer = ImageClassifierTrainer(...,
            callbacks={"terminate", early_stop}
        )
    """

    def __init__(self, monitor: str, min_delta: float = 0., patience: int = 8) -> None:
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False

    def __call__(self, model: nn.Module, epoch: int, valid_metric) -> TrainerState:
        score: float = -valid_metric[self.monitor].avg

        # TODO: rethink about this logic - doesn't seem to do the job.
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        if self.early_stop:
            # TODO: figure out later how and where to save
            # store old metric and save new model
            # torch.save(model, self.filepath)
            print(f"[INFO] Early-Stopping the training process. Epoch: {epoch}.")
            return TrainerState.TERMINATE

        return TrainerState.TRAINING


class ModelCheckpoint:
    """Callback that save the model at the end of everyepoch.

    Args:
        filepath: the where to save the mode.
        monitor: the name of the value to track.

    **Usage example:**

    .. code:: python

        model_checkpoint = ModelCheckpoint(
            filepath="./outputs", monitor="top5",
        )

        trainer = ImageClassifierTrainer(...,
            callbacks={"checkpoint", model_checkpoint}
        )
    """

    def __init__(self, filepath: str, monitor: str) -> None:
        self.filepath = filepath
        self.monitor = monitor

        # track best model
        self.best_metric: float = 0.

        # create directory
        Path(self.filepath).mkdir(parents=True, exist_ok=True)

    def __call__(self, model: nn.Module, epoch: int, valid_metric) -> None:
        valid_metric_value: float = valid_metric[self.monitor].avg
        if valid_metric_value > self.best_metric:
            self.best_metric = valid_metric_value
            # store old metric and save new model
            filename = Path(self.filepath) / f"model_{epoch}.pt"
            torch.save(model, filename)
        ...
