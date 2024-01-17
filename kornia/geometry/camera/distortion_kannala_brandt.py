# inspired by: shttps://github.com/farm-ng/sophus-rs/blob/main/src/sensor/kannala_brandt.rs
import kornia.core as ops
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.camera.distortion_affine import distort_points_affine


def _distort_points_kannala_brandt_impl(
    projected_points_in_camera_z1_plane: Tensor,
    params: Tensor,
) -> Tensor:
    # https://github.com/farm-ng/sophus-rs/blob/20f6cac68f17fe1ac41d0aa8a27489e2b886806f/src/sensor/kannala_brandt.rs#L51-L67
    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    fx, fy = params[..., 0], params[..., 1]
    cx, cy = params[..., 2], params[..., 3]

    k0 = params[..., 4]
    k1 = params[..., 5]
    k2 = params[..., 6]
    k3 = params[..., 7]

    radius_sq = x**2 + y**2

    radius = radius_sq.sqrt()
    radius_inverse = 1.0 / radius
    theta = radius.atan2(ops.ones_like(radius))
    theta2 = theta**2
    theta4 = theta2**2
    theta6 = theta2 * theta4
    theta8 = theta4**2

    r_distorted = theta * (1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8)

    scaling = r_distorted * radius_inverse

    u = fx * scaling * x + cx
    v = fy * scaling * y + cy

    return ops.stack([u, v], dim=-1)


def distort_points_kannala_brandt(projected_points_in_camera_z1_plane: Tensor, params: Tensor) -> Tensor:
    r"""Distort one or more points from the canonical z=1 plane into the camera frame using the Kannala-Brandt
    model.

    Args:
        projected_points_in_camera_z1_plane: Tensor representing the points to distort with shape (..., 2).
        params: Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        Tensor representing the distorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> distort_points_kannala_brandt(points, params)
        tensor([1982.6832, 1526.3619])
    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    radius_sq = x**2 + y**2

    # TODO: we can optimize this by passing the radius_sq to the impl functions. Check if it's worth it.
    distorted_points = ops.where(
        radius_sq[..., None] > 1e-8,
        _distort_points_kannala_brandt_impl(
            projected_points_in_camera_z1_plane,
            params,
        ),
        distort_points_affine(projected_points_in_camera_z1_plane, params[..., :4]),
    )

    return distorted_points


def undistort_points_kannala_brandt(distorted_points_in_camera: Tensor, params: Tensor) -> Tensor:
    r"""Undistort one or more points from the camera frame into the canonical z=1 plane using the Kannala-Brandt
    model.

    Args:
        distorted_points_in_camera: Tensor representing the points to undistort with shape (..., 2).
        params: Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        Tensor representing the undistorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> undistort_points_kannala_brandt(points, params).shape
        torch.Size([2])
    """
    KORNIA_CHECK_SHAPE(distorted_points_in_camera, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    x = distorted_points_in_camera[..., 0]
    y = distorted_points_in_camera[..., 1]

    fx, fy = params[..., 0], params[..., 1]
    cx, cy = params[..., 2], params[..., 3]

    k0 = params[..., 4]
    k1 = params[..., 5]
    k2 = params[..., 6]
    k3 = params[..., 7]

    un = (x - cx) / fx
    vn = (y - cy) / fy
    rth2 = un**2 + vn**2

    # TODO: explore stop condition (won't work with pytorch with batched inputs)
    # Additionally, with this stop condition we can avoid adding 1e-8 to the denominator
    # in the return statement of the function.

    # if rth2.abs() < 1e-8:
    #     return distorted_points_in_camera

    rth = rth2.sqrt()

    th = rth.sqrt()

    iters = 0

    # gauss-newton

    while True:
        th2 = th**2
        th4 = th2**2
        th6 = th2 * th4
        th8 = th4**2

        thd = th * (1.0 + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8)

        d_thd_wtr_th = 1.0 + 3.0 * k0 * th2 + 5.0 * k1 * th4 + 7.0 * k2 * th6 + 9.0 * k3 * th8

        step = (thd - rth) / d_thd_wtr_th
        th = th - step

        iters += 1

        # TODO: improve stop condition by masking only the elements that have converged
        th_abs_mask = th.abs() < 1e-8

        if th_abs_mask.all():
            break

        if iters >= 20:
            break

    radius_undistorted = th.tan()

    undistorted_points = ops.where(
        radius_undistorted[..., None] < 0.0,
        ops.stack(
            [
                -radius_undistorted * un / (rth + 1e-8),
                -radius_undistorted * vn / (rth + 1e-8),
            ],
            dim=-1,
        ),
        ops.stack(
            [
                radius_undistorted * un / (rth + 1e-8),
                radius_undistorted * vn / (rth + 1e-8),
            ],
            dim=-1,
        ),
    )

    return undistorted_points


def dx_distort_points_kannala_brandt(projected_points_in_camera_z1_plane: Tensor, params: Tensor) -> Tensor:
    r"""Compute the derivative of the x distortion with respect to the x coordinate.

    .. math::
        \frac{\partial u}{\partial x} =
        \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}

    Args:
        projected_points_in_camera_z1_plane: Tensor representing the points to distort with shape (..., 2).
        params: Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        Tensor representing the derivative of the x distortion with respect to the x coordinate with shape (..., 2).

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> dx_distort_points_kannala_brandt(points, params)
        tensor([[ 486.0507, -213.5573],
                [-213.5573,  165.7147]])
    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    a = projected_points_in_camera_z1_plane[..., 0]
    b = projected_points_in_camera_z1_plane[..., 1]

    fx, fy = params[..., 0], params[..., 1]

    k0 = params[..., 4]
    k1 = params[..., 5]
    k2 = params[..., 6]
    k3 = params[..., 7]

    # TODO: return identity matrix if a and b are zero
    # radius_sq = a ** 2 + b ** 2

    c0 = a.pow(2.0)
    c1 = b.pow(2.0)
    c2 = c0 + c1
    c3 = c2.pow(5.0 / 2.0)
    c4 = c2 + 1.0
    c5 = c2.sqrt().atan()
    c6 = c5.pow(2.0)
    c7 = c6 * k0
    c8 = c5.pow(4.0)
    c9 = c8 * k1
    c10 = c5.pow(6.0)
    c11 = c10 * k2
    c12 = c5.pow(8.0) * k3
    c13 = 1.0 * c4 * c5 * (c11 + c12 + c7 + c9 + 1.0)
    c14 = c13 * c3
    c15 = c2.pow(3.0 / 2.0)
    c16 = c13 * c15
    c17 = 1.0 * c11 + 1.0 * c12 + 2.0 * c6 * (4.0 * c10 * k3 + 2.0 * c6 * k1 + 3.0 * c8 * k2 + k0)
    c18 = c17 * c2.pow(2.0)
    c19 = 1.0 / c4
    c20 = c19 / c2.pow(3.0)
    c21 = a * b * c19 * (-c13 * c2 + c15 * c17) / c3

    return ops.stack(
        [
            ops.stack([c20 * fx * (-c0 * c16 + c0 * c18 + c14), c21 * fx], dim=-1),
            ops.stack([c21 * fy, c20 * fy * (-c1 * c16 + c1 * c18 + c14)], dim=-1),
        ],
        dim=-2,
    )
