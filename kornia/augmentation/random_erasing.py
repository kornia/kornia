from typing import Tuple

import torch
import torch.nn as nn
import torch.distributions as tdist


class RandomRectangleErasing(nn.Module):
    r"""
    Erases a random selected rectangle for each image in the batch, putting the value to zero.
    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [erase_scale_range[0], erase_scale_range[1]) and an aspect ratio sampled
    between [aspect_ratio_range[0], aspect_ratio_range[1])

    Args:
        erase_scale_range (Tuple[float, float]): range of proportion of erased area against input image.
        aspect_ratio_range (Tuple[float, float]): range of aspect ratio of erased area.

    Examples:
        >>> inputs = torch.ones(1, 1, 3, 3)
        >>> rec_er = kornia.augmentation.RandomRectangleErasing((.4, .8), (.3, 1/.3))
        >>> rec_er(inputs)
        tensor([[[[1., 0., 0.],
                  [1., 0., 0.],
                  [1., 0., 0.]]]])
    """

    def __init__(
            self, erase_scale_range: Tuple[float, float], aspect_ratio_range: Tuple[float, float]
    ) -> None:
        super(RandomRectangleErasing, self).__init__()
        self.erase_scale_range: Tuple[float, float] = erase_scale_range
        self.aspect_ratio_range: Tuple[float, float] = aspect_ratio_range

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore
        return random_rectangle_erase(
            images,
            self.erase_scale_range,
            self.aspect_ratio_range
        )


def get_random_rectangles_params(
        params_shape, images_height, images_width, erase_scale_range, aspect_ratio_range
):
    images_area = images_height * images_width
    target_areas = tdist.Uniform(
        erase_scale_range[0], erase_scale_range[1]
    ).sample(params_shape) * images_area
    if aspect_ratio_range[0] < 1. and aspect_ratio_range[1] > 1.:
        aspect_ratios1 = tdist.Uniform(aspect_ratio_range[0], 1).sample(params_shape)
        aspect_ratios2 = tdist.Uniform(1, aspect_ratio_range[1]).sample(params_shape)
        rand_idxs = torch.round(torch.rand(params_shape)).bool()
        aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
    else:
        aspect_ratios = tdist.Uniform(
            aspect_ratio_range[0], aspect_ratio_range[1]
        ).sample(params_shape)
    # based on target areas and aspect ratios, rectangle params are computed
    heights = torch.min(
        torch.max(torch.round((target_areas * aspect_ratios) ** (1 / 2)), torch.tensor(1.)),
        torch.tensor(float(images_height))
    ).int()
    widths = torch.min(
        torch.max(torch.round((target_areas / aspect_ratios) ** (1 / 2)), torch.tensor(1.)),
        torch.tensor(float(images_width))
    ).int()
    xs = (torch.rand(params_shape) * (images_width - widths + 1).float()).int()
    ys = (torch.rand(params_shape) * (images_height - heights + 1).float()).int()
    return widths, heights, xs, ys


def draw_rectangles(mask_size, rectangle_params, device):
    r"""
    Generate a {0, 1} mask with drawed rectangle having parameters defined by rectangle_params
    and size by mask_size

    Args:
        mask_size (torch.Size or Tuple): output mask size.
        rectangle_params (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
            rectangle_params[0] must be widths tensor
            rectangle_params[1] must be heights tensor
            rectangle_params[2] must be x positions tensor
            rectangle_params[3] must be y positions tensor
        device (torch.device): device to use.
    """
    if not (rectangle_params[0].size() == rectangle_params[1].size(
    ) == rectangle_params[2].size() == rectangle_params[3].size()):
        raise TypeError(
            f"''rectangle params components must have same shape"
        )

    widths, heights, xs, ys = rectangle_params
    mask = torch.zeros(mask_size, dtype=torch.float, device=device)
    for i_elem in range(mask_size[0]):
        h = heights[i_elem].item()
        w = widths[i_elem].item()
        y = ys[i_elem].item()
        x = xs[i_elem].item()
        mask[i_elem, :, y:y + h, x:x + w] = 1.
    return mask


def erase_rectangles(images, rectangle_params):
    device = images.device
    mask = draw_rectangles(images.size(), rectangle_params, device)
    mask = 1. - mask
    return images * mask


def random_rectangle_erase(
        images: torch.Tensor,
        erase_scale_range: Tuple[float, float],
        aspect_ratio_range: Tuple[float, float]
) -> torch.Tensor:
    r"""
    Function that erases a random selected rectangle for each image in the batch, putting
    the value to zero.
    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [erase_scale_range[0], erase_scale_range[1]) and an aspect ratio sampled
    between [aspect_ratio_range[0], aspect_ratio_range[1])

    Args:
        images (torch.Tensor): input images.
        erase_scale_range (Tuple[float, float]): range of proportion of erased area against input image.
        aspect_ratio_range (Tuple[float, float]): range of aspect ratio of erased area.
    """
    if not (isinstance(erase_scale_range[0], float) and isinstance(erase_scale_range[1],
            float) and erase_scale_range[0] > 0. and erase_scale_range[1] > 0.):
        raise TypeError(
            f"'erase_scale_range' must be a Tuple[float, float] with positive values"
        )
    if not (isinstance(aspect_ratio_range[0], float) and isinstance(aspect_ratio_range[1],
            float) and aspect_ratio_range[0] > 0. and aspect_ratio_range[1] > 0.):
        raise TypeError(
            f"'aspect_ratio_range' must be a Tuple[float, float] with positive values"
        )

    images_size = images.size()
    b, _, h, w = images_size
    rect_params = get_random_rectangles_params(
        (b, ), h, w, erase_scale_range, aspect_ratio_range
    )
    images = erase_rectangles(images, rect_params)
    return images
