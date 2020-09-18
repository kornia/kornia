from typing import Optional
import torch

from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    sharpness,
    solarize,
    equalize,
    posterize,
    invert2d,
)
from kornia.geometry import (
    translate,
    shear,
    rotate
)
from ..functional import apply_erase_rectangles
from ..random_generator import random_rectangles_params_generator


def _cutout(input: torch.Tensor, percentage: float = 0.2) -> torch.Tensor:
    r"""Cutout on an image tensor.

    Args:
        percentage (float): range of size of the origin size cropped.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> input = torch.randn(1, 1, 5, 5)
        >>> _cutout(input, 0.4)
        tensor([[[[ 1.2500e+02,  1.2500e+02, -2.5058e-01, -4.3388e-01,  8.4871e-01],
                  [ 1.2500e+02,  1.2500e+02, -2.1152e+00,  3.2227e-01, -1.5771e-01],
                  [ 1.4437e+00,  2.6605e-01,  1.6646e-01,  8.7438e-01, -1.4347e-01],
                  [-1.1161e-01, -6.1358e-01,  1.2590e+00,  2.0050e+00,  5.3737e-02],
                  [ 6.1806e-01, -4.1280e-01, -8.4106e-01, -2.3160e+00, -1.0231e-01]]]])
    """
    batch_size, c, h, w = input.size()
    output = input.clone()
    params = random_rectangles_params_generator(
        input.size(0), h, w, torch.tensor([percentage ** 2, percentage ** 2]),
        torch.tensor([1., 1.]), value=(125, 123, 114) if c == 3 else 125, same_on_batch=True)
    output = apply_erase_rectangles(input, params)
    return output


POLICY_FUNCS = {
    # TODO: Implement AutoContrast
    'autocontrast': lambda input: input,
    'equalize': equalize,
    'invert': invert2d,
    'rotate': lambda inp, angle: rotate(inp, angle, align_corners=True),
    'posterize': posterize,
    'solarize': lambda inp, threshold: solarize(inp, threshold, None),
    'solarizeAdd': lambda inp, additions: solarize(inp, 0.5, additions),
    'color': adjust_saturation,
    'contrast': adjust_contrast,
    'brightness': adjust_brightness,
    'sharpness': sharpness,
    'shearX': lambda inp, shearX: shear(inp, torch.stack([shearX, torch.zeros_like(shearX)], dim=-1), True),
    'shearY': lambda inp, shearY: shear(inp, torch.stack([torch.zeros_like(shearY), shearY], dim=-1), True),
    'translateX': lambda inp, transX: translate(
        inp, torch.stack([transX * inp.size(-2), torch.zeros_like(transX)], dim=-1), True),
    'translateY': lambda inp, transY: translate(
        inp, torch.stack([torch.zeros_like(transY), transY * inp.size(-1)], dim=-1), True),
    'cutout': _cutout,
}


class SubPolicy(object):
    """Subpolicy for auto augmentation."""
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2):
        ranges = {
            "shearX": torch.linspace(-0.3, 0.3, 10),
            "shearY": torch.linspace(-0.3, 0.3, 10),
            "translateX": torch.linspace(-150 / 331, 150 / 331, 10),
            "translateY": torch.linspace(-150 / 331, 150 / 331, 10),
            "rotate": torch.linspace(-30, 30, 10),
            "color": torch.linspace(0.3, 1.0, 10),
            "posterize": torch.round(torch.linspace(8, 4, 10)).to(torch.int),
            "solarize": torch.linspace(1., 0., 10),
            "contrast": torch.linspace(0.3, 1.1, 10),
            "sharpness": torch.linspace(0.1, 0.8, 10),
            "brightness": torch.linspace(-0.6, 0.6, 10),
            "autocontrast": [None] * 10,
            "equalize": [None] * 10,
            "invert": [None] * 10
        }

        self.p1 = p1
        self.operation1 = POLICY_FUNCS[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]

        self.p2 = p2
        self.operation2 = POLICY_FUNCS[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def to_apply(self) -> torch.Tensor:
        return torch.tensor([torch.rand(1).item() < self.p1, torch.rand(1).item() < self.p2])

    def __call__(self, input: torch.Tensor, to_apply: Optional[torch.Tensor] = None):
        if to_apply is None:
            to_apply = self.to_apply()
        if to_apply[0]:
            input = self.operation1(input, self.magnitude1) if self.magnitude1 is not None else self.operation1(input)
        if to_apply[1]:
            input = self.operation2(input, self.magnitude2) if self.magnitude2 is not None else self.operation2(input)
        return input
