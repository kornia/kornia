import torch

from kornia.metrics import accuracy, AverageMeter
from .trainer import Trainer


class ImageClassifierTrainer(Trainer):
    """Module to be used for Image Classification purposes.

    The module subclasses :py:class:`~kornia.x.Trainer` and overrides the
    :py:meth:`~kornia.x.Trainer.evaluate` function implementing a standard accuracy topk@[1, 5].

    .. seealso::
        Learn how to use this class in the following
        `example <https://github.com/kornia/kornia/master/examples/train/train_classifier.py>`_.
    """
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        stats = {'losses': AverageMeter(), 'top1': AverageMeter(), 'top5': AverageMeter()}
        for sample_id, sample in enumerate(self.valid_dataloader):
            source, target = sample  # this might change with new pytorch ataset structure

            # perform the preprocess and augmentations in batch
            img = self.preprocess(source)
            # Forward
            out = self.model(img)
            # Loss computation
            val_loss = self.criterion(out, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(out.detach(), target, topk=(1, 5))
            stats['losses'].update(val_loss.item(), img.shape[0])
            stats['top1'].update(acc1.item(), img.shape[0])
            stats['top5'].update(acc5.item(), img.shape[0])

            if sample_id % 10 == 0:
                self._logger.info(
                    f"Test: {sample_id}/{len(self.valid_dataloader)} "
                    f"Loss: {stats['losses'].val:.2f} {stats['losses'].avg:.2f} "
                    f"Acc@1: {stats['top1'].val:.2f} {stats['top1'].val:.2f} "
                    f"Acc@5: {stats['top5'].val:.2f} {stats['top5'].val:.2f} "
                )

        return stats
