from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kornia.metrics import accuracy, mean_average_precision, mean_iou

from .trainer import Trainer
from .utils import Configuration


class ImageClassifierTrainer(Trainer):
    """Module to be used for image classification purposes.

    The module subclasses :py:class:`~kornia.x.Trainer` and overrides the
    :py:func:`~kornia.x.Trainer.evaluate` function implementing a standard
    :py:func:`~kornia.metrics.accuracy` topk@[1, 5].

    .. seealso::
        Learn how to use this class in the following
        `example <https://github.com/kornia/kornia/blob/master/examples/train/image_classifier/>`__.
    """

    def compute_metrics(self, *args: torch.Tensor) -> dict[str, float]:
        if len(args) != 2:
            raise AssertionError
        out, target = args
        acc1, acc5 = accuracy(out, target, topk=(1, 5))
        return dict(top1=acc1.item(), top5=acc5.item())


class SemanticSegmentationTrainer(Trainer):
    """Module to be used for semantic segmentation purposes.

    The module subclasses :py:class:`~kornia.x.Trainer` and overrides the
    :py:func:`~kornia.x.Trainer.evaluate` function implementing IoU :py:func:`~kornia.metrics.mean_iou`.

    .. seealso::
        Learn how to use this class in the following
        `example <https://github.com/kornia/kornia/blob/master/examples/train/semantic_segmentation/>`__.
    """

    def compute_metrics(self, *args: torch.Tensor) -> dict[str, float]:
        if len(args) != 2:
            raise AssertionError
        out, target = args
        iou = mean_iou(out.argmax(1), target, out.shape[1]).mean()
        return dict(iou=iou.item())


class ObjectDetectionTrainer(Trainer):
    """Module to be used for object detection purposes.

    The module subclasses :py:class:`~kornia.x.Trainer` and overrides the
    :py:func:`~kornia.x.Trainer.evaluate` function implementing IoU :py:func:`~kornia.metrics.mean_iou`.

    .. seealso::
        Learn how to use this class in the following
        `example <https://github.com/kornia/kornia/blob/master/examples/train/object_detection/>`__.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: nn.Module | None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
        config: Configuration,
        num_classes: int,
        callbacks: dict[str, Callable] = None,
        loss_computed_by_model: bool | None = None,
    ) -> None:
        if callbacks is None:
            callbacks = {}
        super().__init__(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, config, callbacks)
        # TODO: auto-detect if the model is from TorchVision
        self.loss_computed_by_model = loss_computed_by_model
        self.num_classes = num_classes

    def on_model(self, model: nn.Module, sample: dict):
        if self.loss_computed_by_model and model.training:
            return model(sample["input"], sample["target"])
        return model(sample["input"])

    def compute_loss(self, *args: torch.Tensor) -> torch.Tensor:
        if self.loss_computed_by_model:
            # Note: in case of dict losses obtained
            if isinstance(args[0], dict):
                return torch.stack([v for _, v in args[0].items()]).mean()
            return torch.stack(list(args[0])).sum()
        if self.criterion is None:
            raise RuntimeError("`criterion` should not be None if `loss_computed_by_model` is False.")
        return self.criterion(*args)

    def compute_metrics(self, *args: torch.Tensor) -> dict[str, float]:
        if (
            isinstance(args[0], dict)
            and "boxes" in args[0]
            and "labels" in args[0]
            and "scores" in args[0]
            and isinstance(args[1], dict)
            and "boxes" in args[1]
            and "labels" in args[1]
        ):
            mAP, _ = mean_average_precision(
                [a['boxes'] for a in args[0]],
                [a['labels'] for a in args[0]],
                [a['scores'] for a in args[0]],
                [a['boxes'] for a in args[1]],
                [a['labels'] for a in args[1]],
                n_classes=self.num_classes,
                threshold=0.000001,
            )
            return {'mAP': mAP.item()}
        return super().compute_metrics(*args)
