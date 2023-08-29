from typing import Union

from kornia.core import Tensor


class AverageMeter:
    """Computes and stores the average and current value.

    Example:
        >>> stats = AverageMeter()
        >>> acc1 = torch.tensor(0.99) # coming from K.metrics.accuracy
        >>> stats.update(acc1, n=1)  # where n is batch size usually
        >>> round(stats.avg, 2)
        0.99
    """

    val: Union[float, bool, Tensor]
    _avg: Union[float, Tensor]
    sum: Union[float, Tensor]
    count: int

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[float, bool, Tensor], n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    @property
    def avg(self) -> float:
        if isinstance(self._avg, Tensor):
            return float(self._avg.item())
        return self._avg
