from typing import Dict, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _common_param_check
from kornia.core import Device, Tensor, tensor
from kornia.geometry.bbox import bbox_generator
from kornia.geometry.transform.affwarp import _side_to_image_size

__all__ = ["ResizeGenerator"]


class ResizeGenerator(RandomGeneratorBase):
    r"""Get parameters for ```resize``` transformation for resize transform.

    Args:
        resize_to: Desired output size of the crop, like (h, w).
        side: Which side to resize if `resize_to` is only of type int.

    Returns:
        parameters to be passed for transformation.
            - src (Tensor): cropping bounding boxes with a shape of (B, 4, 2).
            - dst (Tensor): output bounding boxes with a shape (B, 4, 2).
            - input_size (Tensor): (h, w) from batch input.
            - resize_to (tuple): new (h, w) for batch input.

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, resize_to: Union[int, Tuple[int, int]], side: str = "short") -> None:
        super().__init__()
        self.output_size = resize_to
        self.side = side

    def __repr__(self) -> str:
        repr = f"output_size={self.output_size}"
        return repr

    def make_samplers(self, device: Device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        pass

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device = self.device
        _dtype = self.dtype

        if batch_size == 0:
            return {
                "src": torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
                "dst": torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
            }

        input_size = h, w = (batch_shape[-2], batch_shape[-1])

        src = bbox_generator(
            tensor(0, device=_device, dtype=_dtype),
            tensor(0, device=_device, dtype=_dtype),
            tensor(input_size[1], device=_device, dtype=_dtype),
            tensor(input_size[0], device=_device, dtype=_dtype),
        ).repeat(batch_size, 1, 1)

        if isinstance(self.output_size, int):
            aspect_ratio = w / h
            output_size = _side_to_image_size(self.output_size, aspect_ratio, self.side)
        else:
            output_size = self.output_size

        if not (
            len(output_size) == 2
            and isinstance(output_size[0], (int,))
            and isinstance(output_size[1], (int,))
            and output_size[0] > 0
            and output_size[1] > 0
        ):
            raise AssertionError(f"`resize_to` must be a tuple of 2 positive integers. Got {output_size}.")

        dst = bbox_generator(
            tensor(0, device=_device, dtype=_dtype),
            tensor(0, device=_device, dtype=_dtype),
            tensor(output_size[1], device=_device, dtype=_dtype),
            tensor(output_size[0], device=_device, dtype=_dtype),
        ).repeat(batch_size, 1, 1)

        _input_size = tensor(input_size, device=_device, dtype=torch.long).expand(batch_size, -1)
        _output_size = tensor(output_size, device=_device, dtype=torch.long).expand(batch_size, -1)

        return {"src": src, "dst": dst, "input_size": _input_size, "output_size": _output_size}
