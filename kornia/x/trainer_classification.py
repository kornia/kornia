import torch

from kornia.metrics import accuracy, AverageMeter, mean_iou

from .trainer import Trainer


class ImageClassifierTrainer(Trainer):
    """Module to be used for Image Classification purposes.

    The module subclasses :py:class:`~kornia.x.Trainer` and overrides the
    :py:meth:`~kornia.x.Trainer.evaluate` function implementing a standard accuracy topk@[1, 5].

    .. seealso::
        Learn how to use this class in the following
        `example <https://github.com/kornia/kornia/blob/master/examples/train/image_classifier/>`__.
    """
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        stats = {'losses': AverageMeter(), 'top1': AverageMeter(), 'top5': AverageMeter()}
        for sample_id, sample in enumerate(self.valid_dataloader):
            sample = {"input": sample[0], "target": sample[1]}  # new dataset api will come like this
            # perform the preprocess and augmentations in batch
            sample = self.preprocess(sample)
            # Forward
            out = self.model(sample["input"])
            # Loss computation
            val_loss = self.criterion(out, sample["target"])

            # measure accuracy and record loss
            acc1, acc5 = accuracy(out.detach(), sample["target"], topk=(1, 5))
            batch_size: int = sample["input"].shape[0]
            stats['losses'].update(val_loss.item(), batch_size)
            stats['top1'].update(acc1.item(), batch_size)
            stats['top5'].update(acc5.item(), batch_size)

            if sample_id % 10 == 0:
                self._logger.info(
                    f"Test: {sample_id}/{len(self.valid_dataloader)} "
                    f"Loss: {stats['losses'].val:.2f} {stats['losses'].avg:.2f} "
                    f"Acc@1: {stats['top1'].val:.2f} {stats['top1'].val:.2f} "
                    f"Acc@5: {stats['top5'].val:.2f} {stats['top5'].val:.2f} "
                )

        return stats


class SemanticSegmentationTrainer(Trainer):
    """Module to be used for Semantic segmentation purposes.

    The module subclasses :py:class:`~kornia.x.Trainer` and overrides the
    :py:meth:`~kornia.x.Trainer.evaluate` function implementing IoU :py:meth:`~kornia.utils.mean_iou`.

    .. seealso::
        Learn how to use this class in the following
        `example <https://github.com/kornia/kornia/blob/master/examples/train/semantic_segmentation/>`__.
    """
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        stats = {'losses': AverageMeter(), 'iou': AverageMeter()}
        for sample_id, sample in enumerate(self.valid_dataloader):
            sample = {"input": sample[0], "target": sample[1]}  # new dataset api will come like this
            # perform the preprocess and augmentations in batch
            sample = self.preprocess(sample)
            sample = self.on_before_model(sample)
            # Forward
            out = self.model(sample["input"])
            self.on_after_model(out, sample)
            # Loss computation
            val_loss = self.criterion(out, sample["target"])

            # measure accuracy and record loss
            iou = mean_iou(out.argmax(1), sample["target"], out.shape[1]).mean()
            batch_size: int = sample["input"].shape[0]
            stats['losses'].update(val_loss.item(), batch_size)
            stats['iou'].update(iou, batch_size)

            if sample_id % 10 == 0:
                self._logger.info(
                    f"Test: {sample_id}/{len(self.valid_dataloader)} "
                    f"Loss: {stats['losses'].val:.2f} {stats['losses'].avg:.2f} "
                    f"IoU: {stats['iou'].val:.2f} {stats['iou'].val:.2f} "
                )

        return stats
