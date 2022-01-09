from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def pad_batch_list(data: Sequence[Tensor], padding_value=0.0):
    batch_dim = len(data)
    if batch_dim > 0:
        if isinstance(data, list):

            def calc_dim_pad(stat):
                return tuple(int(item) for size in stat.flip(dims=(0,)) for item in (0, size))

            dims = torch.Tensor([d.shape for d in data])
            max_dims = torch.amax(dims, dim=0)
            padding_dims = torch.stack([max_dims - torch.Tensor(list(d.shape)) for d in data])
            padded = torch.stack(
                [F.pad(d, calc_dim_pad(padding_dims[i]), value=padding_value) for i, d in enumerate(data)]
            )
            return padded, padding_dims
    return data, torch.empty(0)


class AugmentedTensor(torch.Tensor):
    """Augmented Tensor, based on https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/rnn.py."""

    @staticmethod
    def __new__(cls, x, device=None, padding_value=0.0, *args, **kwargs):
        if isinstance(x, torch.Tensor) and (device is not None or x.device != "cpu"):
            device = x.device if device is None else device

        if isinstance(x, Tensor):
            tensor = super().__new__(cls, x, *args, **kwargs)
            padding_dims = [[0] * len(x.shape[1:])] * x.shape[0]
        else:
            padded, padding_dims = pad_batch_list(x, padding_value=padding_value)
            tensor = super().__new__(cls, padded, *args, **kwargs)

        if device is not None:
            tensor = tensor.to(device)

        tensor.padding_dims = padding_dims.tolist()

        return tensor

    _padding_dims: Sequence[int]

    @property
    def padding_dims(self):
        return self._padding_dims

    @padding_dims.setter
    def padding_dims(self, dims):
        self._padding_dims = dims

    def __init__(self, x, **kwargs):
        super().__init__()

    def clone(self, *args, **kwargs):
        return AugmentedTensor(super().clone(*args, **kwargs), self._padding_dims)

    def to(self, *args, **kwargs):
        new_obj = AugmentedTensor([], self._padding_dims)
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        return new_obj

    def __repr__(self):
        properties = [super().__repr__(), f"padding_dims={self._padding_dims.__repr__()}"]
        return ", ".join(properties)


at = AugmentedTensor([])
at = AugmentedTensor([torch.Tensor([[1, 2, 3]]), torch.Tensor([[1, 2], [3, 4]])])
