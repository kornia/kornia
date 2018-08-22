import torch


def convert_points_from_homogeneous(points, eps=1e-6):
    """Converts points from homogeneous to Euclidean space.

    Args:
        points (Tensor): tensor of N-dimensional points of size (B, D, N).

    Returns:
        Tensor: tensor of N-1-dimensional points of size (B, D, N-1).
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(points)))

    if len(points.shape) != 3:
        raise ValueError("Input size must be a three dimensional tensor. Got {}"
                         .format(points.shape))

    return points[..., :-1] / (points[..., -1:] + eps)


def convert_points_to_homogeneous(points):
    """Converts points from Euclidean to homogeneous space.

    Args:
        points (Tensor): tensor of N-dimensional points of size (B, D, N).

    Returns:
        Tensor: tensor of N+1-dimensional points of size (B, D, N+1).
    """
    if not torch.is_tensor(points):
        raise TypeError("Input ype is not a torch.Tensor. Got {}"
                        .format(type(points)))

    if len(points.shape) != 3:
        raise ValueError("Input size must be a three dimensional tensor. Got {}"
                         .format(points.shape))

    return torch.cat([points, torch.ones_like(points)[..., :1]], dim=-1)
