import cv2
import numpy as np
import torch
import torchgeometry as tgm

__all__ = [
    "tensor_to_image",
    "image_to_tensor",
    "draw_rectangle",
    "create_pinhole"
]


def image_to_tensor(image):
    """Converts a numpy image to a torch.Tensor image.
    """
    # TODO: add asserts and type checkings
    tensor = torch.from_numpy(image)
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.permute(2, 0, 1)  # CxHxW


def tensor_to_image(tensor):
    """Converts a torch.Tensor image to a numpy image. In case the tensor is
       in the GPU, it will be copied back to CPU.
    """
    # TODO: add asserts and type checkings
    tensor = torch.squeeze(tensor)
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
