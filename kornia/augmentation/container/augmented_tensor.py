from typing import List, Sequence, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch._C import dtype


def pad_batch_list(data: Sequence[Tensor], padding_value=0.0):
    batch_dim = len(data)
    if batch_dim > 0:
        if isinstance(data, list):

            def calc_dim_pad(stat):
                return tuple(int(item) for size in stat.flip(dims=(0,)) for item in (0, size))

            dims = torch.IntTensor([d.shape for d in data])
            max_dims = torch.amax(dims, dim=0)
            padding_dims = torch.stack([max_dims - torch.IntTensor(list(d.shape)) for d in data])
            padded = torch.stack(
                [F.pad(d, calc_dim_pad(padding_dims[i]), value=padding_value) for i, d in enumerate(data)]
            )
            return padded, padding_dims
    return data, torch.empty(0)


class AugmentedTensor(torch.Tensor):
    """Augmented Tensor is a simple, limited attempt to support a list of tensors as input for batch processing
    untill a ragged/nested `Tensor` exists in PyTorch.

    Args:
        x: The input data. Input needs to be tensor of (B, N, ...) or list of (N, ...).
        padding_value: The padding value. Depends on data type of input and will not perform any checks.
        Always pads to the right.

    Returns:
        The batched output tensor, shape of :math:`(B, N, ...)`.

    Note:
        :math:`(...)` denotes abritraty dimensions.

        Will be replaced by `NestedTensor` (`https://github.com/pytorch/nestedtensor`) as soon as it is merged.
        Includes ideas from `https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/rnn.py`.

    Examples:
        >>> at = AugmentedTensor([])
        >>> at = AugmentedTensor([
        ...         torch.Tensor([
        ...             [1, 2, 3],
        ...         ]),
        ...         torch.Tensor([
        ...             [1, 2],
        ...             [3, 4]
        ...         ])
        ...     ])  # ((1x3), (2x2))

        >>> at
        tensor([[[0., 0., 0., 0., 0.],
                 [0., 1., 1., 1., 1.],
                 [0., 1., 1., 1., 1.],
                 [0., 1., 1., 1., 1.],
                 [0., 0., 0., 0., 0.]]])
    """

    @staticmethod
    def __new__(cls, x: Union[Tensor, List[Tensor]], padding_value: Union[float, int, bool] = 0.0, *args, **kwargs):
        padding_dims = kwargs.pop("padding_dims", [])
        if isinstance(x, AugmentedTensor):
            tensor = super().__new__(cls, x, *args, **kwargs)
        elif isinstance(x, Tensor):
            tensor = super().__new__(cls, x, *args, **kwargs)
            padding_dims = [[0] * len(x.shape[1:])] * x.shape[0]
        else:
            padded, padding_dims = pad_batch_list(x, padding_value=padding_value)
            padding_dims = padding_dims.tolist()
            tensor = super().__new__(cls, padded, *args, **kwargs)

        tensor._padding_dims = padding_dims

        return tensor

    _padding_dims: Sequence[int]

    def __init__(self, x, **kwargs):
        super().__init__()

    def clone(self, *args, **kwargs):
        return AugmentedTensor(super().clone(*args, **kwargs), padding_dims=self._padding_dims)

    def tolist(self, dim=None) -> List[Tensor]:
        if dim is None:
            return super().tolist()
        return torch.chunk(self, self.shape[dim], dim=dim)

    def to(self, *args, **kwargs):
        new_obj = AugmentedTensor([])
        new_obj._padding_dims = self._padding_dims
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        return new_obj
