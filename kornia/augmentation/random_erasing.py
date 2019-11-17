from typing import Tuple

import torch
import torch.nn as nn
import torch.distributions as tdist


class RandomRectangleErasing(nn.Module):
    r"""
    Erases a random selected rectangle in the batch tensor images, putting the value to zero.
    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [erase_scale[0], erase_scale[1]) and an aspect ratio sampled between
    [aspect_ratio[0], aspect_ratio[1])

    Args:
        erase_scale (Tuple[float, float]): range of proportion of erased area against input image.
        aspect_ratio (Tuple[float, float]): range of aspect ratio of erased area.

    Examples:
        >>> input = torch.ones(1, 1, 3, 3)
        >>> rec_er = kornia.augmentation.RandomRectangleErasing((.4, .8), (.3, 1/.3))
        >>> rec_er(input)
        tensor([[[[1., 1., 1.],
                  [0., 0., 1.],
                  [0., 0., 1.]]]])
    """

    def __init__(
            self, erase_scale: Tuple[float, float], aspect_ratio: Tuple[float, float]
    ) -> None:
        super(RandomRectangleErasing, self).__init__()
        assert erase_scale[0] > 0.
        assert aspect_ratio[0] > 0.

        self.erase_scale: Tuple[float, float] = erase_scale
        self.aspect_ratio: Tuple[float, float] = aspect_ratio

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore
        b, c, h, w = images.size()
        images_area = w * h

        target_areas = tdist.Uniform(
            self.erase_scale[0], self.erase_scale[1]
        ).sample((b,)) * images_area

        if self.aspect_ratio[0] < 1. and self.aspect_ratio[1] > 1.:
            aspect_ratios1 = tdist.Uniform(self.aspect_ratio[0], 1).sample((b,))
            aspect_ratios2 = tdist.Uniform(1, self.aspect_ratio[1]).sample((b,))
            rand_idxs = torch.round(torch.rand((b,))).bool()
            aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
        else:
            aspect_ratios = tdist.Uniform(self.aspect_ratio[0], self.aspect_ratio[1]).sample((b,))
        # compute the mask defining the rectangles
        heights = torch.min(
            torch.max(torch.round((target_areas * aspect_ratios) ** (1 / 2)), torch.tensor(1.)),
            torch.tensor(float(h))
        ).int()
        widths = torch.min(
            torch.max(torch.round((target_areas / aspect_ratios) ** (1 / 2)), torch.tensor(1.)),
            torch.tensor(float(w))
        ).int()
        masks = torch.ones_like(images)
        for i_elem in range(b):
            pos_x = torch.randint(0, w - widths[i_elem] + 1, (1,)).item()
            pos_y = torch.randint(0, h - heights[i_elem] + 1, (1,)).item()
            pos_x = int(pos_x)
            pos_y = int(pos_y)
            new_h = int(heights[i_elem])
            new_w = int(widths[i_elem])
            masks[i_elem, :, pos_y:pos_y + new_h, pos_x:pos_x + new_w] = 0.

        return images * masks


def random_rectangle_erase(
        images: torch.Tensor,
        erase_scale: Tuple[float, float],
        aspect_ratio: Tuple[float, float]) -> torch.Tensor:
    r"""
    Function that erases a random selected rectangle in the batch tensor images, putting
    the value to zero.

    See :class:`~kornia.augmentation.RandomRectangleErasing` for details.
    """
    return RandomRectangleErasing(erase_scale, aspect_ratio)(images)
