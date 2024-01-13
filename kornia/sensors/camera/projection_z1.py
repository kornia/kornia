import torch

from kornia.core import Tensor


def project_points_z1(points_in_camera: Tensor) -> Tensor:
    """Project one or more points from the camera frame into the canonical z=1 plane through perspective division.

    Project one or more 3-points from the camera frame into the canonical
    z=1 plane through perspective division. For N points, a 3xN matrix must be
    provided where each column is a point to be transformed. The result will
    be a 2xN matrix. N may be dynamically sized, but the input columns must be
    statically determined as 3 at compile time.

    Args:
        points_in_camera: Tensor representing the points to project.

    Returns:
        Tensor representing the projected points.

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> project_points_z1(points)
        tensor([0.3333, 0.6667])
    """
    return points_in_camera[:2] / (points_in_camera[2] + 1e-08)


def unproject_points_z1(points_in_cam_canonical: Tensor, extension: Tensor) -> Tensor:
    unpprojected = points_in_cam_canonical * extension
    return torch.pad(unpprojected, (0, 1), "constant", 1.0)


def dx_proj_x(points_in_camera: Tensor) -> Tensor:
    """Compute the derivative of the x projection with respect to the x coordinate.

    Args:
        points_in_camera: Tensor representing the points to project.

    Returns:
        Tensor representing the derivative of the x projection with respect to the x coordinate.

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> dx_proj_x(points)
        tensor([0.3333, 0.0000, 0.0000])
    """
    x = points_in_camera[..., 0]
    y = points_in_camera[..., 1]
    z = points_in_camera[..., 2]

    z_inv = 1.0 / (z + 1e-08)
    z_sq = z_inv * z_inv
    zeros = torch.zeros_like(z_inv)
    return torch.stack(
        [
            torch.concatenate([z_inv, zeros, -x * z_sq], dim=-1),
            torch.concatenate([zeros, z_inv, -y * z_sq], dim=-1),
        ],
        dim=-1,
    )
