"""Module containing differentiable functions for Brown-Conrady distortion model."""
# inspired by: https://github.com/farm-ng/farm-ng-core/blob/e872fbe1aebd26fabfabefd77054ef9774301d67/cpp/sophus/sensor/camera_distortion/brown_conrady.h
import kornia.core as ops
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.camera.distortion_affine import distort_points_affine, undistort_points_affine


# from: https://github.com/opencv/opencv/blob/63bb2abadab875fc648a572faccafee134f06fc8/modules/calib3d/src/calibration.cpp#L791
def _project_points_brown_conrady_impl(points_normalized: Tensor, distortion_params: Tensor) -> Tensor:
    x = points_normalized[..., 0]
    y = points_normalized[..., 1]

    k0 = distortion_params[..., 0]
    k1 = distortion_params[..., 1]
    k2 = distortion_params[..., 2]
    k3 = distortion_params[..., 3]
    k4 = distortion_params[..., 4]
    k5 = distortion_params[..., 5]
    k6 = distortion_params[..., 6]
    k7 = distortion_params[..., 7]

    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2 * r4
    a1 = 2.0 * x * y
    a2 = r2 + 2.0 * x**2
    a3 = r2 + 2.0 * y**2

    cdist = 1.0 + k0 * r2 + k1 * r4 + k4 * r6
    icdist2 = 1.0 / (1.0 + k5 * r2 + k6 * r4 + k7 * r6)

    xd0 = x * cdist * icdist2 + k2 * a1 + k3 * a2
    yd0 = y * cdist * icdist2 + k2 * a3 + k3 * a1

    return ops.stack([xd0, yd0], dim=-1)


def _unproject_points_brown_conrady_impl(uv_normalized: Tensor, distortion_params: Tensor) -> Tensor:
    # the distortion parameters coefficients
    d0 = distortion_params[..., 0]
    d1 = distortion_params[..., 1]
    d2 = distortion_params[..., 2]
    d3 = distortion_params[..., 3]
    d4 = distortion_params[..., 4]
    d5 = distortion_params[..., 5]
    d6 = distortion_params[..., 6]
    d7 = distortion_params[..., 7]

    # initial guess
    xy = uv_normalized.clone()

    for _ in range(50):
        f_xy = _project_points_brown_conrady_impl(xy, distortion_params) - uv_normalized
        f_xy = f_xy[..., None]  # (..., 2, 1)

        # generate jacobian
        a = xy[..., 0]  # x
        b = xy[..., 1]  # y

        c0 = a * a  # pow(a, 2);
        c1 = b * b  # pow(b, 2);
        c2 = c0 + c1
        c3 = c2 * c2  # pow(c2, 2);
        c4 = c3 * c2  # pow(c2, 3);
        c5 = c2 * d5 + c3 * d6 + c4 * d7 + 1.0
        c6 = c5 * c5  # pow(c5, 2);
        c7 = 1.0 / c6
        c8 = a * d3
        c9 = 2.0 * d2
        c10 = 2.0 * c2
        c11 = 3.0 * c3
        c12 = c2 * d0
        c13 = c3 * d1
        c14 = c4 * d4
        c15 = 2.0 * (c10 * d6 + c11 * d7 + d5) * (c12 + c13 + c14 + 1.0)
        c16 = 2.0 * c10 * d1 + 2.0 * c11 * d4 + 2.0 * d0
        c17 = 1.0 * c12 + 1.0 * c13 + 1.0 * c14 + 1.0
        c18 = b * d3
        c19 = a * b
        c20 = -c15 * c19 + c16 * c19 * c5
        du_dx = c7 * (-c0 * c15 + c5 * (c0 * c16 + c17) + c6 * (b * c9 + 6.0 * c8))
        du_dy = c7 * (c20 + c6 * (a * c9 + 2.0 * c18))
        dv_dx = c7 * (c20 + c6 * (2 * a * d2 + 2.0 * c18))
        dv_dy = c7 * (-c1 * c15 + c5 * (c1 * c16 + c17) + c6 * (6.0 * b * d2 + 2.0 * c8))

        #     | du_dx  du_dy |      | a  b |
        # J = |              |  =:  |      |
        #     | dv_dx  dv_dy |      | c  d |

        A = du_dx
        B = du_dy
        C = dv_dx
        D = dv_dy

        # | a  b | -1       1   |  d  -b |
        # |      |     =  ----- |        |
        # | c  d |        ad-bc | -c   a |

        # m = ops.stack(
        #    [
        #        ops.stack([+D, -B], dim=-1),
        #        ops.stack([-C, +A], dim=-1),
        #    ],
        #    dim=-2,
        # )

        # det = A * D - B * C
        # j_inv = (1.0 / det)[..., None, None] * m # (..., 2, 2)
        # j_inv = 1.0 / ((A * D - B * C)[..., None, None] * m + 1e-12) # (..., 2, 2)
        M = ops.stack(
            [
                ops.stack([A, B], dim=-1),
                ops.stack([C, D], dim=-1),
            ],
            dim=-2,
        )

        j_inv = M.inverse()  # (..., 2, 2)
        step_i = j_inv @ f_xy  # (..., 2, 1)

        xy -= step_i[..., 0]  # (..., 2)
        print(xy)

    return xy


def distort_points_brown_conrady(projected_points_in_camera_z1_plane: Tensor, params: Tensor) -> Tensor:
    r"""Distort one or more points from the canonical z=1 plane into the camera frame using the Brown-Conrady
    distortion model.

    Args:
        projected_points_in_camera_z1_plane: Tensor representing the points to distort with shape (..., 2).
        params: Tensor representing the parameters of the Brown-Conrady distortion model with shape (..., 5).

    Returns:
        Tensor representing the distorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> params = torch.tensor([600., 600., 319.5, 239.5, 0.1])
        >>> distort_points_brown_conrady(points, params)
        tensor([1369.8710, 2340.2419])
    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "12"])

    distorted_point_in_camera_z1_plane = _project_points_brown_conrady_impl(
        projected_points_in_camera_z1_plane,
        params[..., 4:],
    )

    return distort_points_affine(distorted_point_in_camera_z1_plane, params[..., :4])


def undistort_points_brown_conrady(projected_points_in_camera_z1_plane: Tensor, params: Tensor) -> Tensor:
    r"""Undistort one or more points from the canonical z=1 plane into the camera frame using the Brown-Conrady
    distortion model.

    Args:
        projected_points_in_camera_z1_plane: Tensor representing the points to undistort with shape (..., 2).
        params: Tensor representing the parameters of the Brown-Conrady distortion model with shape (..., 5).

    Returns:
        Tensor representing the undistorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([1369.8710, 2340.2419])
        >>> params = torch.tensor([600., 600., 319.5, 239.5, 0.1])
        >>> undistort_points_brown_conrady(points, params)
        tensor([1.0000, 2.0000])
    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "12"])

    proj_point_in_camera_z1_plane = _unproject_points_brown_conrady_impl(
        undistort_points_affine(projected_points_in_camera_z1_plane, params[..., :4]), params[..., 4:]
    )

    return proj_point_in_camera_z1_plane
