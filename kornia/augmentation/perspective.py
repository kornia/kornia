from typing import Union, Tuple, cast

import torch
import torch.nn as nn

import kornia


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _get_perspective_params(
    batch_size: int,
    width: int, height: int,
    distortion_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        batch_size (int): the tensor batch size.
        width (int): width of the image.
        height (int) : height of the image.
        distortion_scale (float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

    Returns:
        List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        The points are in -x order.
    """
    start_points: torch.Tensor = torch.tensor([[
        [0., 0],
        [0, width - 1],
        [height - 1, 0],
        [height - 1, width - 1],
    ]]).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx: float = distortion_scale * width / 2
    fy: float = distortion_scale * height / 2

    factor = torch.tensor([fy, fx]).view(-1, 1, 2)
    offset = factor * torch.rand(batch_size, 4, 2) * 2 - 1

    end_points = start_points + offset

    return start_points, end_points


def perspective(input: torch.Tensor,
                start_points: torch.Tensor,
                end_points: torch.Tensor,
                return_transform: bool = False) -> UnionType:
    r"""Perform perspective transform of the given torch.Tensor or batch of tensors.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape BxCxHxW.
        start_points (torch.Tensor): Tensor containing [top-left, top-right, bottom-right,
        bottom-left] of the orignal image with shape Bx4x2.
        end_points (torch.Tensor): Tensor containing [top-left, top-right, bottom-right,
        bottom-left] of the transformed image with shape Bx4x2.
        return_transform (bool): if ``True`` return the matrix describing the transformation
        applied to each. Default: False.

    Returns:
        torch.Tensor:  Perspectively transformed tensor.
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not torch.is_tensor(start_points):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(start_points)}")

    if not torch.is_tensor(end_points):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(end_points)}")

    # compute the homography between the input points
    transform: torch.Tensor = kornia.get_perspective_transform(start_points, end_points)

    # apply the computed transform
    height, width = input.shape[-2:]
    img_warped: torch.Tensor = kornia.warp_perspective(input, transform, (height, width))

    if return_transform:
        return img_warped, transform

    return img_warped


def random_perspective(input: torch.Tensor,
                       distortion_scale: float = 0.5,
                       p: float = 0.5,
                       return_transform: bool = False) -> UnionType:
    r"""Performs Perspective transformation of the given torch.Tensor randomly with a given probability.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape (*, C, H, W).
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
        return_transform (bool): if ``True`` return the matrix describing the transformation
        applied to each. Default: False.
        input tensor.
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) > 4:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    # arrange input data
    x_data: torch.Tensor = input.view(-1, *input.shape[-3:])

    batch_size, _, height, width = x_data.shape

    # sample probabilities
    probs: torch.Tensor = torch.rand(batch_size, device=x_data.device)
    mask: torch.Tensor = probs < p

    # compute points
    start_points, end_points = (
        _get_perspective_params(batch_size, width, height, distortion_scale)
    )
    start_points = start_points.to(x_data.device, x_data.dtype)
    end_points = end_points.to(x_data.device, x_data.dtype)

    # compute and apply transform
    out_perspective: UnionType = perspective(
        x_data, start_points, end_points, return_transform
    )

    out_data: torch.Tensor = x_data.clone()

    if return_transform:
        x_warped, transform = out_perspective
        out_data[mask] = x_warped[mask]
        return out_data, transform
    else:
        x_warped = cast(torch.Tensor, out_perspective)

    out_data[mask] = x_warped[mask]

    return out_data


class RandomPerspective(nn.Module):
    r"""Performs Perspective transformation of the given torch.Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
        return_transform (bool): if ``True`` return the matrix describing the transformation
        applied to each. Default: False.
        input tensor.
    """
    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomPerspective, self).__init__()
        self.p: float = p
        self.distortion_scale: float = distortion_scale
        self.return_transform: bool = return_transform

    def __repr__(self) -> str:
        repr = f"(distortion_scale={self.distortion_scale}, p={self.p}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    def forward(self, input: UnionType) -> UnionType:  # type: ignore

        if isinstance(input, tuple):
            raise NotImplementedError("wait for AugmentationBase class")
        return random_perspective(
            input, distortion_scale=self.distortion_scale, p=self.p,
            return_transform=self.return_transform)
