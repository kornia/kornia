from typing import Any, Dict, Optional, Tuple

from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._pc.intensity.base import IntensityAugmentationBasePC
from kornia.augmentation.utils import _range_bound


class RandomScalePC(IntensityAugmentationBasePC):
    r"""Scales a given point cloud.

    Args:
        scale_range: scaling range.
        axes: the axes to jitter with.
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        same_on_batch: apply the same transformation across the batch.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        axes: str = "xyz",
        same_on_batch: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch)

        self.scale_range: Tensor = _range_bound(scale_range, 'scale', bounds=None)
        self._param_generator = rg.PlainUniformGenerator((self.scale_range, "scale_factor", None, None))
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
        input = input.clone()
        input[..., self._dims] = input[..., self._dims] * params["scale_factor"]
        return input
