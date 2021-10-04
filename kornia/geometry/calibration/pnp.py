import torch
from kornia.geometry.conversions import convert_points_to_homogeneous


def solve_pnp_dlt(
    world_points: torch.Tensor, img_points: torch.Tensor,
    intrinsic: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    r"""This function attempts to solve the Perspective-n-Point (PnP)
    problem using Direct Linear Transform (DLT).

    Given :math:`N` 3D points (where :math:`N \geq 6`) in the world
    space, :math:`N` corresponding 2D points in the image space and
    an intrinsic matrix, this function tries to estimate the world
    to camera transformation matrix.

    This function needs atleast 6 points (i.e. :math:`N \geq 6`) to
    provide a solution. This function cannot be used if all the 3D world
    points lie on a line or if all the 3D world points lie on a plane.

    Args:
        world_points        :   A tensor with shape :math:`(N, 3)` representing
                                the points in the world space.
        img_points          :   A tensor with shape :math:`(N, 2)` representing
                                the points in the image space.
        intrinsic           :   A tensor with shape :math:`(3, 3)` representing
                                the intrinsic matrix.

    Returns:
        A tensor with shape :math:`(3, 4)` representing the estimated world to
        camera transformation matrix.
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

    if type(intrinsic) is not torch.Tensor:
        raise TypeError(f"Type of intrinsic is not torch.Tensor. Got {type(intrinsic)}")

    if type(eps) is not float:
        raise TypeError(f"Type of eps is not float. Got {type(world_points)}")

    if (len(world_points.shape) == 2) and (world_points.shape[1] != 3):
        raise AssertionError(
            f"world_points must be of shape (N, 3). Got shape {world_points.shape}."
        )

    if (len(img_points.shape) == 2) and (img_points.shape[1] != 2):
        raise AssertionError(
            f"img_points must be of shape (N, 2). Got shape {img_points.shape}."
        )

    if intrinsic.shape != (3, 3):
        raise AssertionError(
            f"intrinsic must be of shape (3, 3). Got shape {intrinsic.shape}."
        )

    if world_points.shape[0] != img_points.shape[0]:
        raise AssertionError("world_points and img_points must have equal number of points.")

    if world_points.shape[0] < 6:
        raise AssertionError(
            f"Atleast 6 points are required to use this function. "
            f"Got {world_points.shape[0]} points."
        )

    # Checking if world_points has rank = 3. This function cannot be used
    # if all world_points lie on a line or if all world points lie on a plane.
    s = torch.linalg.svdvals(world_points)
    if s[-1] < eps:
        raise AssertionError(
            f"The last singular value is smaller than {eps}. This function "
            f"cannot be used if all world_points lie on a line "
            f"or if all world_points lie on a plane."
        )

    N = world_points.shape[0]
    intrinsic_inv = torch.linalg.inv(intrinsic)

    img_points_h = convert_points_to_homogeneous(img_points)
    world_points_h = convert_points_to_homogeneous(world_points)

    # Getting normalized image points.
    norm_points = torch.matmul(intrinsic_inv, img_points_h.T).T

    # Setting up the system (the matrix A in Ax=0)
    system = torch.zeros((2 * N, 12), dtype=world_points.dtype, device=world_points.device)
    system[0::2, 0:4] = world_points_h
    system[1::2, 4:8] = world_points_h
    system[0::2, 8:12] = world_points_h * (-1) * norm_points[:, 0:1]
    system[1::2, 8:12] = world_points_h * (-1) * norm_points[:, 1:2]

    # Getting the solution vector.
    u, s, v = torch.svd(system)
    solution = v[:, -1]

    # Reshaping the solution vector to the correct shape.
    pred_world_to_cam = solution.reshape(3, 4)

    # We may need to multiply pred_world_to_cam with a scalar. This is
    # because if x is a solution to Ax=0, then cx is also a solution.
    # We can find this scalar by using the properties of the
    # rotation matrix. We do this in two parts:

    # First, we fix the sign by making sure that the determinant of
    # the rotation matrix is non negative (since determinant of a rotation
    # matrix should be 1).
    det = torch.linalg.det(pred_world_to_cam[:3, :3])
    if det < 0:
        pred_world_to_cam = pred_world_to_cam * -1

    # Then, we make sure that norm of a column of the rotation matrix
    # is 1. Do note that the norm of any column of a rotation matrix
    # should be 1. Here we use the 0th column to calculate norm_col.
    # We then multiply all elements of pred_world_to_cam with (1/norm_col)
    norm_col = torch.linalg.norm(pred_world_to_cam[:3, 0])
    pred_world_to_cam = pred_world_to_cam * (1 / norm_col)

    return pred_world_to_cam
