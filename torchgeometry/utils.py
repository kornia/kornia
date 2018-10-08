import torch
import torch.nn as nn
import torchgeometry as tgm

import numpy as np


__all__ = [
    "tensor_to_image",
    "image_to_tensor",
    "draw_rectangle",
    "create_pinhole",
    "create_meshgrid",
    "inverse",
    "Inverse",
]


def create_meshgrid(height, width, normalized_coordinates=True):
    """Generates a coordinate grid for an image.

    This is normalized to be in the range [-1,1] to be consistent with the
    pytorch function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): wether to used normalized coordinates in
        the range [-1, 1] in order to be consistent with the PyTorch function
        grid_sample.

    Return:
        Tensor: returns a 1xHxWx2 grid.
    """
    # generate coordinates
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxwx2


def inverse(transforms):
    """Batched version of `torch.inverse`

    Args:
        transforms (Tensor): tensor of transformations of size (B, D, D).

    Returns:
        Tensor: tensor of inverted transformations of size (B, D, D).

    """
    if not len(transforms.shape) == 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                transforms.shape))
    # iterate, compute inverse and stack tensors
    return torch.stack([torch.inverse(transform) for transform in transforms])


class Inverse(nn.Module):
    def __init__(self):
        super(Inverse, self).__init__()

    def forward(self, input):
        return inverse(input)


def image_to_tensor(image):
    """Converts a numpy image to a torch.Tensor image.

    Args:
        image (numpy.ndarray): image of the form (H, W, C).

    Returns:
        numpy.ndarray: image of the form (H, W, C).

    """
    if not type(image) == np.ndarray:
        raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
            type(image)))
    if len(image.shape) > 3 or len(image.shape) < 2:
        raise ValueError("Input size must be a two or three dimensional array")
    tensor = torch.from_numpy(image)
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.permute(2, 0, 1)  # CxHxW


def tensor_to_image(tensor):
    """Converts a torch.Tensor image to a numpy image. In case the tensor is in
       the GPU, it will be copied back to CPU.

    Args:
        tensor (Tensor): image of the form (1, C, H, W).

    Returns:
        numpy.ndarray: image of the form (H, W, C).

    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))
    tensor = torch.squeeze(tensor)
    if len(tensor.shape) > 3 or len(tensor.shape) < 2:
        raise ValueError(
            "Input size must be a two or three dimensional tensor")
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.permute(1, 2, 0).contiguous().cpu().detach().numpy()


def create_pinhole(intrinsic, extrinsic, height, width):
    pinhole = torch.zeros(12)
    pinhole[0] = intrinsic[0, 0]  # fx
    pinhole[1] = intrinsic[1, 1]  # fy
    pinhole[2] = intrinsic[0, 2]  # cx
    pinhole[3] = intrinsic[1, 2]  # cy
    pinhole[4] = height
    pinhole[5] = width
    pinhole[6:9] = tgm.rotation_matrix_to_angle_axis(
        torch.tensor(extrinsic))
    pinhole[9:12] = torch.tensor(extrinsic[:, 3])
    return pinhole.view(1, -1)


def draw_rectangle(image, dst_homo_src):
    import cv2
    height, width = image.shape[:2]
    pts_src = torch.FloatTensor([[
        [-1, -1],  # top-left
        [1, -1],  # bottom-left
        [1, 1],  # bottom-right
        [-1, 1],  # top-right
    ]]).to(dst_homo_src.device)
    # transform points
    pts_dst = tgm.transform_points(tgm.inverse(dst_homo_src), pts_src)

    def compute_factor(size):
        return 1.0 * size / 2

    def convert_coordinates_to_pixel(coordinates, factor):
        return factor * (coordinates + 1.0)
    # compute convertion factor
    x_factor = compute_factor(width - 1)
    y_factor = compute_factor(height - 1)
    pts_dst = pts_dst.cpu().squeeze().detach().numpy()
    pts_dst[..., 0] = convert_coordinates_to_pixel(
        pts_dst[..., 0], x_factor)
    pts_dst[..., 1] = convert_coordinates_to_pixel(
        pts_dst[..., 1], y_factor)
    # do the actual drawing
    for i in range(4):
        pt_i, pt_ii = tuple(pts_dst[i % 4]), tuple(pts_dst[(i + 1) % 4])
        image = cv2.line(image, pt_i, pt_ii, (255, 0, 0), 3)
    return image
