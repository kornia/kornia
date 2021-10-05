import torch

import kornia
from kornia.geometry.conversions import convert_points_to_homogeneous


def solve_pnp_dlt(
    world_points: torch.Tensor, img_points: torch.Tensor,
    intrinsics: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    r"""This function attempts to solve the Perspective-n-Point (PnP)
    problem using Direct Linear Transform (DLT).

    Given a batch (where batch size is :math:`B`) of :math:`N` 3D points
    (where :math:`N \geq 6`) in the world space, a batch of :math:`N`
    corresponding 2D points in the image space and a batch of
    intrinsic matrices, this function tries to estimate a batch of
    world to camera transformation matrices.

    This implementation needs atleast 6 points (i.e. :math:`N \geq 6`) to
    provide solutions. This function cannot be used if all the 3D world
    points (of any element of the batch) lie on a line or if all the
    3D world points (of any element of the batch) lie on a plane.

    Args:
        world_points        :   A tensor with shape :math:`(B, N, 3)` representing
                                the points in the world space.
        img_points          :   A tensor with shape :math:`(B, N, 2)` representing
                                the points in the image space.
        intrinsics          :   A tensor with shape :math:`(B, 3, 3)` representing
                                the intrinsic matrices.

    Returns:
        A tensor with shape :math:`(B, 3, 4)` representing the estimated world to
        camera transformation matrices.
    """
    # This function was implemented based on ideas inspired from multiple references.
    # ============
    # References:
    # ============
    # 1. https://team.inria.fr/lagadic/camera_localization/tutorial-pose-dlt-opencv.html
    # 2. https://github.com/opencv/opencv/blob/68d15fc62edad980f1ffa15ee478438335f39cc3/modules/calib3d/src/calibration.cpp # noqa: E501
    # 3. http://rpg.ifi.uzh.ch/docs/teaching/2020/03_camera_calibration.pdf
    # 4. http://www.cs.cmu.edu/~16385/s17/Slides/11.3_Pose_Estimation.pdf

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
            f"Atleast 6 points are required to use this function. "
            f"Got {world_points.shape[1]} points."
        )

    # Checking if world_points (of any element of the batch) has rank = 3. This
    # function cannot be used if all world_points (of any element of the batch) lie
    # on a line or if all world points lie on a plane.
    _, s, _ = torch.svd(world_points)
    if torch.any(s[:, -1] < eps):
        raise AssertionError(
            f"The last singular value is smaller than {eps}. This function "
            f"cannot be used if all world_points (of any element of the batch) "
            f"lie on a line or if all world_points (of any element of the batch) "
            f"lie on a plane."
        )

    B, N = world_points.shape[:2]
    intrinsics_inv = torch.inverse(intrinsics)
    world_points_h = convert_points_to_homogeneous(world_points)

    # Getting normalized image points.
    norm_points = kornia.geometry.transform_points(intrinsics_inv, img_points)

    # Setting up the system (the matrix A in Ax=0)
    system = torch.zeros((B, 2 * N, 12), dtype=world_points.dtype, device=world_points.device)
    system[:, 0::2, 0:4] = world_points_h
    system[:, 1::2, 4:8] = world_points_h
    system[:, 0::2, 8:12] = world_points_h * (-1) * norm_points[..., 0:1]
    system[:, 1::2, 8:12] = world_points_h * (-1) * norm_points[..., 1:2]

    # Getting the solution vectors.
    _, _, v = torch.svd(system)
    solution = v[..., -1]

    # Reshaping the solution vectors to the correct shape.
    pred_world_to_cam = solution.reshape(B, 3, 4)

    # We obtained one solution for each element of the batch. We may
    # need to multiply each solution with a scalar. This is because
    # if x is a solution to Ax=0, then cx is also a solution. We can
    # find the required scalars by using the properties of
    # rotation matrices. We do this in two parts:

    # First, we fix the sign by making sure that the determinant of
    # the all the rotation matrices are non negative (since determinant
    # of a rotation matrix should be 1).
    det = torch.det(pred_world_to_cam[:, :3, :3])
    sign_fix = torch.where(det < 0, -1, 1)
    pred_world_to_cam = pred_world_to_cam * sign_fix[:, None, None]

    # Then, we make sure that norm of the 0th columns of the rotation
    # matrices are 1. Do note that the norm of any column of a rotation
    # matrix should be 1. Here we use the 0th column to calculate norm_col.
    # We then multiply pred_world_to_cam with mul_factor.
    norm_col = torch.sqrt(torch.sum(pred_world_to_cam[:, :3, 0] ** 2, axis=1))
    mul_factor = (1 / norm_col)[:, None, None]
    pred_world_to_cam = pred_world_to_cam * mul_factor

    return pred_world_to_cam
