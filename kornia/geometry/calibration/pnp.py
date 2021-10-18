from typing import Optional, Tuple

import torch

import kornia
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.linalg import transform_points

__all__ = [
    "solve_pnp_dlt",
]


def _mean_isotropic_scale_normalize(
    points: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Normalizes points.

    Args:
       points : Tensor containing the points to be normalized with shape :math:`(B, N, D)`.
       eps : Small value to avoid division by zero error.

    Returns:
       Tuple containing the normalized points in the shape :math:`(B, N, D)` and the transformation matrix
       in the shape :math:`(B, D+1, D+1)`.

    """
    if not isinstance(points, torch.Tensor):
        raise AssertionError(f"points is not an instance of torch.Tensor. Type of points is {type(points)}")

    if len(points.shape) != 3:
        raise AssertionError(f"points must be of shape (B, N, D). Got shape {points.shape}.")

    x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1xD
    scale = (points - x_mean).norm(dim=-1, p=2).mean(dim=-1)  # B

    D_int = points.shape[-1]
    D_float = torch.tensor(points.shape[-1], dtype=torch.float64, device=points.device)
    scale = torch.sqrt(D_float) / (scale + eps)  # B
    transform = kornia.eye_like(D_int + 1, points)  # (B, D+1, D+1)

    idxs = torch.arange(D_int, dtype=torch.int64, device=points.device)
    transform[:, idxs, idxs] = transform[:, idxs, idxs] * scale[:, None]
    transform[:, idxs, D_int] = transform[:, idxs, D_int] + (-scale[:, None] * x_mean[:, 0, idxs])

    points_norm = transform_points(transform, points)  # BxNxD

    return (points_norm, transform)


def solve_pnp_dlt(
    world_points: torch.Tensor, img_points: torch.Tensor,
    intrinsics: torch.Tensor, weights: Optional[torch.Tensor] = None,
    svd_eps: float = 1e-4,
) -> torch.Tensor:
    r"""This function attempts to solve the Perspective-n-Point (PnP)
    problem using Direct Linear Transform (DLT).

    Given a batch (where batch size is :math:`B`) of :math:`N` 3D points
    (where :math:`N \geq 6`) in the world space, a batch of :math:`N`
    corresponding 2D points in the image space and a batch of
    intrinsic matrices, this function tries to estimate a batch of
    world to camera transformation matrices.

    This implementation needs at least 6 points (i.e. :math:`N \geq 6`) to
    provide solutions.

    This function cannot be used if all the 3D world points (of any element
    of the batch) lie on a line or if all the 3D world points (of any element
    of the batch) lie on a plane. This function attempts to check for these
    conditions and throws an AssertionError if found. Do note that this check
    is sensitive to the value of the svd_eps parameter.

    Another bad condition occurs when the camera and the points lie on a
    twisted cubic. However, this function does not check for this condition.

    Args:
        world_points : A tensor with shape :math:`(B, N, 3)` representing
          the points in the world space.
        img_points : A tensor with shape :math:`(B, N, 2)` representing
          the points in the image space.
        intrinsics : A tensor with shape :math:`(B, 3, 3)` representing
          the intrinsic matrices.
        weights : This parameter is not used currently and is just a
          placeholder for API consistency.
        svd_eps : A small float value to avoid numerical precision issues.

    Returns:
        A tensor with shape :math:`(B, 3, 4)` representing the estimated world to
        camera transformation matrices (also known as the extrinsic matrices).

    Example:
        >>> world_points = torch.tensor([[
        ...     [ -12.7270,  -89.8532,  -63.0994], [ -12.1998,  -83.9055,  -64.7339],
        ...     [ -15.1191, -120.8419,  -52.0851], [ -13.4082, -137.2118,  -42.9774],
        ...     [ -13.8576,  -95.0873,  -61.2726], [ -10.7982,  -82.9056,  -62.0320],
        ... ]], dtype=torch.float32)
        >>>
        >>> img_points = torch.tensor([[
        ...     [ 25.,  32.], [ 61.,  21.], [ 45.,  60.],
        ...     [ 90.,  67.], [ 12.,  65.], [540., 250.],
        ... ]], dtype=torch.float32)
        >>>
        >>> intrinsics = torch.tensor([[
        ...     [ 500.,    0.,  250.],
        ...     [   0.,  500.,  250.],
        ...     [   0.,    0.,    1.],
        ... ]], dtype=torch.float32)
        >>>
        >>> print(world_points.shape, img_points.shape, intrinsics.shape)
        torch.Size([1, 6, 3]) torch.Size([1, 6, 2]) torch.Size([1, 3, 3])
        >>>
        >>> pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)
        >>>
        >>> print(pred_world_to_cam.shape)
        torch.Size([1, 3, 4])
        >>>
        >>> print(pred_world_to_cam)
        tensor([[[ 7.4932e-01,  4.7886e-01,  4.5750e-01,  7.8143e+01],
                 [-6.6168e-01,  5.6720e-01,  4.8917e-01,  7.0223e+01],
                 [-2.6304e-02, -6.6945e-01,  7.4265e-01, -6.3175e+00]]])
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

    if not isinstance(world_points, torch.Tensor):
        raise AssertionError(
            f"world_points is not an instance of torch.Tensor. Type of world_points is {type(world_points)}"
        )

    if not isinstance(img_points, torch.Tensor):
        raise AssertionError(
            f"img_points is not an instance of torch.Tensor. Type of img_points is {type(img_points)}"
        )

    if not isinstance(intrinsics, torch.Tensor):
        raise AssertionError(
            f"intrinsics is not an instance of torch.Tensor. Type of intrinsics is {type(intrinsics)}"
        )

    if (weights is not None) and (not isinstance(weights, torch.Tensor)):
        raise AssertionError(
            f"If weights is not None, then weights should be an instance "
            f"of torch.Tensor. Type of weights is {type(weights)}"
        )

    if type(svd_eps) is not float:
        raise AssertionError(f"Type of svd_eps is not float. Got {type(svd_eps)}")

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

    B, N = world_points.shape[:2]

    # Getting normalized world points.
    world_points_norm, world_transform_norm = _mean_isotropic_scale_normalize(world_points)

    # Checking if world_points_norm (of any element of the batch) has rank = 3. This
    # function cannot be used if all world points (of any element of the batch) lie
    # on a line or if all world points (of any element of the batch) lie on a plane.
    _, s, _ = torch.svd(world_points_norm)
    if torch.any(s[:, -1] < svd_eps):
        raise AssertionError(
            f"The last singular value of one/more of the elements of the batch is smaller "
            f"than {svd_eps}. This function cannot be used if all world_points (of any "
            f"element of the batch) lie on a line or if all world_points (of any "
            f"element of the batch) lie on a plane."
        )

    intrinsics_inv = torch.inverse(intrinsics)
    world_points_norm_h = convert_points_to_homogeneous(world_points_norm)

    # Transforming img_points with intrinsics_inv to get img_points_inv
    img_points_inv = transform_points(intrinsics_inv, img_points)

    # Normalizing img_points_inv
    img_points_norm, img_transform_norm = _mean_isotropic_scale_normalize(img_points_inv)
    inv_img_transform_norm = torch.inverse(img_transform_norm)

    # Setting up the system (the matrix A in Ax=0)
    system = torch.zeros((B, 2 * N, 12), dtype=world_points.dtype, device=world_points.device)
    system[:, 0::2, 0:4] = world_points_norm_h
    system[:, 1::2, 4:8] = world_points_norm_h
    system[:, 0::2, 8:12] = world_points_norm_h * (-1) * img_points_norm[..., 0:1]
    system[:, 1::2, 8:12] = world_points_norm_h * (-1) * img_points_norm[..., 1:2]

    # Getting the solution vectors.
    _, _, v = torch.svd(system)
    solution = v[..., -1]

    # Reshaping the solution vectors to the correct shape.
    solution = solution.reshape(B, 3, 4)

    # Creating solution_4x4
    solution_4x4 = kornia.eye_like(4, solution)
    solution_4x4[:, :3, :] = solution

    # De-normalizing the solution
    intermediate = torch.bmm(solution_4x4, world_transform_norm)
    solution = torch.bmm(inv_img_transform_norm, intermediate[:, :3, :])

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
    norm_col = torch.norm(input=solution[:, :3, 0], p=2, dim=1)
    mul_factor = (1 / norm_col)[:, None, None]
    pred_world_to_cam = solution * mul_factor

    # TODO: Implement algorithm to refine the solution.

    return pred_world_to_cam
