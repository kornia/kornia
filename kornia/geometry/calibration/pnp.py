import torch

import kornia
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.linalg import transform_points, compose_transformations

from typing import Tuple, Optional


def mean_isotropic_scale_normalize(
    points: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Normalizes points.

    Args:
       points : Tensor containing the points to be normalized with shape :math:`(B, N, D)`.
       eps : Small value to avoid division by zero error.

    Returns:
       tuple containing the normalized points in the shape :math:`(B, N, D)` and the transformation matrix
       in the shape :math:`(B, D+1, D+1)`.

    """
    if len(points.shape) != 3:
        raise AssertionError(points.shape)

    x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1xD
    scale = (points - x_mean).norm(dim=-1, p=2).mean(dim=-1)  # B

    D_int = points.shape[-1]
    D_float = torch.tensor(points.shape[-1], dtype=torch.float64, device=points.device)
    scale = torch.sqrt(D_float) / (scale + eps)  # B
    transform = kornia.eye_like(D_int + 1, points)  # (B, D+1, D+1)

    idxs = torch.arange(D_int, dtype=torch.int64, device=points.device)
    transform[:, idxs, idxs] = transform[:, idxs, idxs] * scale[:, None]
    transform[:, idxs, D_int] = transform[:, idxs, D_int] + (-scale[:, None] * x_mean[:, 0, idxs])

    points_norm = kornia.transform_points(transform, points)  # BxNxD

    return (points_norm, transform)


def solve_pnp_dlt(
    world_points: torch.Tensor, img_points: torch.Tensor,
    intrinsics: torch.Tensor, weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""This function attempts to solve the Perspective-n-Point (PnP)
    problem using Direct Linear Transform (DLT).

    Given a batch (where batch size is :math:`B`) of :math:`N` 3D points
    (where :math:`N \geq 6`) in the world space, a batch of :math:`N`
    corresponding 2D points in the image space and a batch of
    intrinsic matrices, this function tries to estimate a batch of
    world to camera transformation matrices.

    This implementation needs at least 6 points (i.e. :math:`N \geq 6`) to
    provide solutions. This function cannot be used if all the 3D world
    points (of any element of the batch) lie on a line or if all the
    3D world points (of any element of the batch) lie on a plane.

    Args:
        world_points : A tensor with shape :math:`(B, N, 3)` representing
          the points in the world space.
        img_points : A tensor with shape :math:`(B, N, 2)` representing
          the points in the image space.
        intrinsics : A tensor with shape :math:`(B, 3, 3)` representing
          the intrinsic matrices.
        weights : This parameter is not used currently and is just a
          placeholder for API consistency.
        eps : A small float value to avoid numerical precision issues
          and division by zero errors.

    Returns:
        A tensor with shape :math:`(B, 3, 4)` representing the estimated world to
        camera transformation matrices (also known as the extrinsic matrices).
    """
    # This function was implemented based on ideas inspired from multiple references.
    # ============
    # References:
    # ============
    # 1. https://team.inria.fr/lagadic/camera_localization/tutorial-pose-dlt-opencv.html
    # 2. https://github.com/opencv/opencv/blob/68d15fc62edad980f1ffa15ee478438335f39cc3/modules/calib3d/src/calibration.cpp # noqa: E501
    # 3. http://rpg.ifi.uzh.ch/docs/teaching/2020/03_camera_calibration.pdf
    # 4. http://www.cs.cmu.edu/~16385/s17/Slides/11.3_Pose_Estimation.pdf
    # 5. https://www.ece.mcmaster.ca/~shirani/vision/hartley_ch7.pdf

    if type(world_points) is not torch.Tensor:
        raise TypeError(f"Type of world_points is not torch.Tensor. Got {type(world_points)}")

    if type(img_points) is not torch.Tensor:
        raise TypeError(f"Type of img_points is not torch.Tensor. Got {type(img_points)}")

    if type(intrinsics) is not torch.Tensor:
        raise TypeError(f"Type of intrinsics is not torch.Tensor. Got {type(intrinsics)}")

    if type(eps) is not float:
        raise TypeError(f"Type of eps is not float. Got {type(world_points)}")

    if (len(world_points.shape) != 3) or (world_points.shape[2] != 3):
        raise AssertionError(
            f"world_points must be of shape (B, N, 3). Got shape {world_points.shape}."
        )

    if (len(img_points.shape) != 3) or (img_points.shape[2] != 2):
        raise AssertionError(
            f"img_points must be of shape (B, N, 2). Got shape {img_points.shape}."
        )

    if (len(intrinsics.shape) != 3) or (intrinsics.shape[1:] != (3, 3)):
        raise AssertionError(
            f"intrinsics must be of shape (B, 3, 3). Got shape {intrinsics.shape}."
        )

    if world_points.shape[1] != img_points.shape[1]:
        raise AssertionError("world_points and img_points must have equal number of points.")

    if (world_points.shape[0] != img_points.shape[0]) or (world_points.shape[0] != intrinsics.shape[0]):
        raise AssertionError("world_points, img_points and intrinsics must have the same batch size.")

    if world_points.shape[1] < 6:
        raise AssertionError(
            f"At least 6 points are required to use this function. "
            f"Got {world_points.shape[1]} points."
        )

    # torch.set_printoptions(precision=6, sci_mode=False)
    B, N = world_points.shape[:2]

    # Getting normalized world points.
    norm_world_points, norm_world_transform = mean_isotropic_scale_normalize(world_points)

    # Checking if world_points (of any element of the batch) has rank = 3. This
    # function cannot be used if all world_points (of any element of the batch) lie
    # on a line or if all world points (of any element of the batch) lie on a plane.
    _, s, _ = torch.svd(norm_world_points)
    if torch.any(s[:, -1] < eps):
        raise AssertionError(
            f"The last singular value is smaller than {eps}. This function "
            f"cannot be used if all world_points (of any element of the batch) "
            f"lie on a line or if all world_points (of any element of the batch) "
            f"lie on a plane."
        )

    intrinsics_inv = torch.inverse(intrinsics)
    norm_world_points_h = convert_points_to_homogeneous(norm_world_points)

    # Transforming img_points with intrinsics_inv to get img_points_
    img_points_ = transform_points(intrinsics_inv, img_points)
    # Normalizing img_points_
    norm_img_points, norm_img_transform = mean_isotropic_scale_normalize(img_points_)
    inv_norm_img_transform = torch.inverse(norm_img_transform)

    # Setting up the system (the matrix A in Ax=0)
    system = torch.zeros((B, 2 * N, 12), dtype=world_points.dtype, device=world_points.device)
    system[:, 0::2, 0:4] = norm_world_points_h
    system[:, 1::2, 4:8] = norm_world_points_h
    system[:, 0::2, 8:12] = norm_world_points_h * (-1) * norm_img_points[..., 0:1]
    system[:, 1::2, 8:12] = norm_world_points_h * (-1) * norm_img_points[..., 1:2]

    # Getting the solution vectors.
    _, _, v = torch.svd(system)
    solution = v[..., -1]

    # Reshaping the solution vectors to the correct shape.
    solution = solution.reshape(B, 3, 4)

    # We obtained one solution for each element of the batch. We may
    # need to multiply each solution with a scalar. This is because
    # if x is a solution to Ax=0, then cx is also a solution. We can
    # find the required scalars by using the properties of
    # rotation matrices. We do this in two parts:

    # First, we fix the sign by making sure that the determinant of
    # the all the rotation matrices are non negative (since determinant
    # of a rotation matrix should be 1).
    det = torch.det(solution[:, :3, :3])
    ones = torch.ones_like(det)
    sign_fix = torch.where(det < 0, ones * -1, ones)
    solution = solution * sign_fix[:, None, None]

    # Then, we make sure that norm of the 0th columns of the rotation
    # matrices are 1. Do note that the norm of any column of a rotation
    # matrix should be 1. Here we use the 0th column to calculate norm_col.
    # We then multiply solution with mul_factor.
    norm_col = torch.sqrt(torch.sum(input=solution[:, :3, 0] ** 2, dim=1))
    mul_factor = (1 / (norm_col + eps))[:, None, None]
    solution = solution * mul_factor

    solution_4x4 = kornia.eye_like(4, solution)
    solution_4x4[:, :3, :] = solution

    # De-normalizing the solution
    temp = torch.bmm(solution_4x4, norm_world_transform)
    pred_world_to_cam = torch.bmm(inv_norm_img_transform, temp[:, :3, :])

    # Fixing the scale of pred_world_to_cam
    norm_col = torch.sqrt(torch.sum(input=pred_world_to_cam[:, :3, 0] ** 2, dim=1))
    mul_factor = (1 / (norm_col + eps))[:, None, None]
    pred_world_to_cam = pred_world_to_cam * mul_factor

    # TODO: Implement algorithm to refine pred_world_to_cam

    return pred_world_to_cam
