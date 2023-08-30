from typing import Any, Dict, Optional

from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._pc.intensity.base import IntensityAugmentationBasePC


class RandomJitterPC(IntensityAugmentationBasePC):
    r"""Shifts the coordinates of a given point cloud.

    Args:
        jitter_scale: standard deviation for Gaussian distribution.
        axes: the axes to jitter with.
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def __init__(
        self,
        jitter_scale: float = 0.01,
        axes: str = "xyz",
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        self.jitter_scale = jitter_scale
        self._param_generator = rg.JitteringGeneratorPC(jitter_scale)
        self.axes = axes
        self._dims = []
        for axis in self.axes:
            if axis == "x":
                self._dims.append(0)
            elif axis == "y":
                self._dims.append(0)
            elif axis == "z":
                self._dims.append(0)
            else:
                raise ValueError(f"Axis `{axis}` not found.")

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        input[..., self._dims] = input[..., self._dims] + params["jitter"][..., self._dims]
        return input


class RandomRGBJitterPC(IntensityAugmentationBasePC):
    r"""Shifts the RGB color of a given point cloud.

    Args:
        jitter_scale: standard deviation for Gaussian distribution.
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def __init__(
        self, jitter_scale: float = 0.1, same_on_batch: bool = False, p: float = 1.0, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        self.jitter_scale = jitter_scale
        self._param_generator = rg.JitteringGeneratorPC(jitter_scale)

    def validate_tensor(self, input: Tensor) -> None:
        """Check if the input tensor is formatted as expected."""
        super().validate_tensor(input)
        if input.size(-1) >= 9:
            raise RuntimeError(f"Expect C >= 9 for including RGB values. Got {input.shape}.")

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        input[..., 6:9] = input[..., 6:9] + params["jitter"]
        return input
