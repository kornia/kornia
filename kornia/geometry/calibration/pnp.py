from typing import Optional, Tuple

import torch
from torch.linalg import qr as linalg_qr

from kornia.core import arange, ones_like, where, zeros
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.geometry.conversions import convert_points_to_homogeneous, normalize_points_with_intrinsics
from kornia.geometry.linalg import transform_points
from kornia.utils import eye_like
from kornia.utils.helpers import _torch_linalg_svdvals, _torch_svd_cast


def _mean_isotropic_scale_normalize(points: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Normalizes points.

    Args:
       points : Tensor containing the points to be normalized with shape :math:`(B, N, D)`.
       eps : Small value to avoid division by zero error.

    Returns:
       Tuple containing the normalized points in the shape :math:`(B, N, D)` and the transformation matrix
       in the shape :math:`(B, D+1, D+1)`.
    """
    KORNIA_CHECK_SHAPE(points, ["B", "N", "D"])
    x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1xD
    scale = (points - x_mean).norm(dim=-1, p=2).mean(dim=-1)  # B

    D_int = points.shape[-1]
    D_float = torch.tensor(points.shape[-1], dtype=torch.float64, device=points.device)
    scale = torch.sqrt(D_float) / (scale + eps)  # B
    transform = eye_like(D_int + 1, points)  # (B, D+1, D+1)

    idxs = arange(D_int, dtype=torch.int64, device=points.device)
    transform[:, idxs, idxs] = transform[:, idxs, idxs] * scale[:, None]
    transform[:, idxs, D_int] = transform[:, idxs, D_int] + (-scale[:, None] * x_mean[:, 0, idxs])

    points_norm = transform_points(transform, points)  # BxNxD

    return (points_norm, transform)


def solve_pnp_dlt(
    world_points: torch.Tensor,
    img_points: torch.Tensor,
    intrinsics: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    svd_eps: float = 1e-4,
) -> torch.Tensor:
    r"""This function attempts to solve the Perspective-n-Point (PnP) problem using Direct Linear Transform (DLT).

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
        weights : A tensor with shape :math:`(B, N)` representing the
            weights for each point. If None, all points are considered to be equally important.
        svd_eps : A small float value to avoid numerical precision issues.

    Returns:
        A tensor with shape :math:`(B, 3, 4)` representing the estimated world to
        camera transformation matrices (also known as the extrinsic matrices).

    Example:
        >>> world_points = torch.tensor([[
        ...     [ 5. , -5. ,  0. ], [ 0. ,  0. ,  1.5],
        ...     [ 2.5,  3. ,  6. ], [ 9. , -2. ,  3. ],
        ...     [-4. ,  5. ,  2. ], [-5. ,  5. ,  1. ],
        ... ]], dtype=torch.float64)
        >>>
        >>> img_points = torch.tensor([[
        ...     [1409.1504, -800.936 ], [ 407.0207, -182.1229],
        ...     [ 392.7021,  177.9428], [1016.838 ,   -2.9416],
        ...     [ -63.1116,  142.9204], [-219.3874,   99.666 ],
        ... ]], dtype=torch.float64)
        >>>
        >>> intrinsics = torch.tensor([[
        ...     [ 500.,    0.,  250.],
        ...     [   0.,  500.,  250.],
        ...     [   0.,    0.,    1.],
        ... ]], dtype=torch.float64)
        >>>
        >>> print(world_points.shape, img_points.shape, intrinsics.shape)
        torch.Size([1, 6, 3]) torch.Size([1, 6, 2]) torch.Size([1, 3, 3])
        >>>
        >>> pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)
        >>>
        >>> print(pred_world_to_cam.shape)
        torch.Size([1, 3, 4])
        >>>
        >>> pred_world_to_cam
        tensor([[[ 0.9392, -0.3432, -0.0130,  1.6734],
                 [ 0.3390,  0.9324, -0.1254, -4.3634],
                 [ 0.0552,  0.1134,  0.9920,  3.7785]]], dtype=torch.float64)
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
    accepted_dtypes = (torch.float32, torch.float64)
    KORNIA_CHECK_IS_TENSOR(world_points)
    KORNIA_CHECK_IS_TENSOR(img_points)
    KORNIA_CHECK_IS_TENSOR(intrinsics)
    KORNIA_CHECK(isinstance(svd_eps, float))
    KORNIA_CHECK(world_points.dtype in accepted_dtypes)
    KORNIA_CHECK(img_points.dtype in accepted_dtypes)
    KORNIA_CHECK(intrinsics.dtype in accepted_dtypes)
    KORNIA_CHECK_SHAPE(world_points, ["B", "N", "3"])
    KORNIA_CHECK_SHAPE(img_points, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(intrinsics, ["B", "3", "3"])
    KORNIA_CHECK_SAME_SHAPE(world_points[:, :, 0], img_points[:, :, 0])
    KORNIA_CHECK(world_points.shape[1] >= 6)
    if weights is not None:
        KORNIA_CHECK_IS_TENSOR(weights)

    B, N = world_points.shape[:2]

    # Getting normalized world points.
    world_points_norm, world_transform_norm = _mean_isotropic_scale_normalize(world_points)

    # Checking if world_points_norm (of any element of the batch) has rank = 3. This
    # function cannot be used if all world points (of any element of the batch) lie
    # on a line or if all world points (of any element of the batch) lie on a plane.
    s = _torch_linalg_svdvals(world_points_norm)
    if torch.any(s[:, -1] < svd_eps):
        raise AssertionError(
            "The last singular value of one/more of the elements of the batch is smaller "
            f"than {svd_eps}. This function cannot be used if all world_points (of any "
            "element of the batch) lie on a line or if all world_points (of any "
            "element of the batch) lie on a plane."
        )
    world_points_norm_h = convert_points_to_homogeneous(world_points_norm)

    # Normalizing img_points_inv
    img_points_inv = normalize_points_with_intrinsics(img_points, intrinsics)
    img_points_norm, img_transform_norm = _mean_isotropic_scale_normalize(img_points_inv)
    inv_img_transform_norm = torch.inverse(img_transform_norm)

    # Setting up the system (the matrix A in Ax=0)
    system = zeros((B, 2 * N, 12), dtype=world_points.dtype, device=world_points.device)
    system[:, 0::2, 0:4] = world_points_norm_h
    system[:, 1::2, 4:8] = world_points_norm_h
    system[:, 0::2, 8:12] = world_points_norm_h * (-1) * img_points_norm[..., 0:1]
    system[:, 1::2, 8:12] = world_points_norm_h * (-1) * img_points_norm[..., 1:2]

    # Apply weights to the system if provided
    if weights is not None:
        if weights.shape != (B, N):
            raise AssertionError(f"Weights should have shape (B, N). Got {weights.shape}.")
        weights_expanded = weights.unsqueeze(-1).repeat(1, 1, 2).view(B, 2 * N, 1)
        # Multiply the system matrix by the expanded weights
        system = weights_expanded * system

    # Getting the solution vectors.
    _, _, v = _torch_svd_cast(system)
    solution = v[..., -1]

    # Reshaping the solution vectors to the correct shape.
    solution = solution.reshape(B, 3, 4)

    # Creating solution_4x4
    solution_4x4 = eye_like(4, solution)
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
    # the all the rotation matrices are non-negative (since determinant
    # of a rotation matrix should be 1).
    det = torch.det(solution[:, :3, :3])
    ones = ones_like(det)
    sign_fix = where(det < 0, ones * -1, ones)
    solution = solution * sign_fix[:, None, None]

    # Then, we make sure that norm of the 0th columns of the rotation
    # matrices are 1. Do note that the norm of any column of a rotation
    # matrix should be 1. Here we use the 0th column to calculate norm_col.
    # We then multiply solution with mul_factor.
    norm_col = torch.norm(input=solution[:, :3, 0], p=2, dim=1)
    mul_factor = (1 / norm_col)[:, None, None]
    temp = solution * mul_factor

    # To make sure that the rotation matrix would be orthogonal, we apply
    # QR decomposition.
    ortho, right = linalg_qr(temp[:, :3, :3])

    # We may need to fix the signs of the columns of the ortho matrix.
    # If right[i, j, j] is negative, then we need to flip the signs of
    # the column ortho[i, :, j]. The below code performs the necessary
    # operations in a better way.
    mask = eye_like(3, ortho)
    col_sign_fix = torch.sign(mask * right)
    rot_mat = torch.bmm(ortho, col_sign_fix)

    # Preparing the final output.
    pred_world_to_cam = torch.cat([rot_mat, temp[:, :3, 3:4]], dim=-1)

    # TODO: Implement algorithm to refine the solution.

    return pred_world_to_cam
