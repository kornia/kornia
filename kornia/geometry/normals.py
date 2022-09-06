import torch
import torch.nn.functional as F

from kornia.core import Tensor
from kornia.geometry.depth import depth_to_3d
from kornia.testing import KORNIA_CHECK_SHAPE


__all__ = [
    "normals_from_depth_accurate",
    "normals_from_depth_forward_gradient",
    "normals_from_depth_improved",
    "normals_from_depth_least_squares",
]


def normals_from_depth_accurate(
    depth: Tensor,
    camera_matrix: Tensor,
    *,
    normalize_points: bool = False,
) -> Tensor:
    r"""Compute the normal surface per pixel.
    Based on: https://atyuwen.github.io/posts/normal-reconstruction/#fn:2

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
            represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> normals_from_depth_accurate(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    KORNIA_CHECK_SHAPE(depth, ["B", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

    # compute the 3d points from depth
    xyz: Tensor = depth_to_3d(depth, camera_matrix, normalize_points)  # Bx3xHxW

    xyz_padded: Tensor = F.pad(xyz, [2, 2, 2, 2], mode="constant")

    # horizontal taps
    xyzh_padded = xyz_padded[..., 2:-2, :]
    h_x, h_z = xyzh_padded[..., 1:-3], xyzh_padded[..., :-4]
    h_y, h_w = xyzh_padded[..., 3:-1], xyzh_padded[..., 4:]
    he_x: Tensor = ((2.0 * h_x[:, 2:3, :, :] - h_z[:, 2:3, :, :]) - depth).abs()
    he_y: Tensor = ((2.0 * h_y[:, 2:3, :, :] - h_w[:, 2:3, :, :]) - depth).abs()

    # vertical taps
    xyzv_padded = xyz_padded[..., 2:-2]
    v_x, v_z = xyzv_padded[..., 1:-3, :], xyzv_padded[..., :-4, :]
    v_y, v_w = xyzv_padded[..., 3:-1, :], xyzv_padded[..., 4:, :]
    ve_x: Tensor = ((2.0 * v_x[:, 2:3, :, :] - v_z[:, 2:3, :, :]) - depth).abs()
    ve_y: Tensor = ((2.0 * v_y[:, 2:3, :, :] - v_w[:, 2:3, :, :]) - depth).abs()

    # create masks:
    #  he_x > he_y - Calculate horizontal derivative of world position from taps | * | y |
    #  he_x < he_y - Calculate horizontal derivative of world position from taps | x | * |
    #  ve_x > ve_y - Calculate vertical derivative of world position from taps | * | y |
    #  ve_x < ve_y - Calculate vertical derivative of world position from taps | x | * |
    h_mask: Tensor = (he_x < he_y).type_as(depth)
    v_mask: Tensor = (ve_x < ve_y).type_as(depth)
    ddh: Tensor = h_mask * (h_x - h_z) + (1 - h_mask) * (h_w - h_y)
    ddv: Tensor = v_mask * (v_x - v_z) + (1 - v_mask) * (v_w - v_y)

    normals: Tensor = torch.cross(ddh, ddv, dim=1)
    return F.normalize(normals, dim=1, p=2)


def normals_from_depth_forward_gradient(
    depth: Tensor,
    camera_matrix: Tensor,
    *,
    normalize_points: bool = False,
) -> Tensor:
    r"""Compute the normal surface per pixel.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
            represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> normals_from_depth_forward_gradient(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    KORNIA_CHECK_SHAPE(depth, ["B", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

    # compute the 3d points from depth
    xyz: Tensor = depth_to_3d(depth, camera_matrix, normalize_points)  # Bx3xHxW

    xyz_padded_x: Tensor = F.pad(xyz, (0, 1, 0, 0), mode="replicate")
    xyz_padded_y: Tensor = F.pad(xyz, (0, 0, 0, 1), mode="replicate")

    grad_x: Tensor = xyz_padded_x[:, :, :, :-1] - xyz_padded_x[:, :, :, 1:]
    grad_y: Tensor = xyz_padded_y[:, :, :-1, :] - xyz_padded_y[:, :, 1:, :]

    normals: Tensor = torch.cross(grad_x, grad_y, dim=1)
    return F.normalize(normals, dim=1, p=2)


def normals_from_depth_improved(
    depth: Tensor,
    camera_matrix: Tensor,
    *,
    normalize_points: bool = False,
) -> Tensor:
    r"""Compute the normal surface per pixel.
    Based on: https://wickedengine.net/2019/09/22/improved-normal-reconstruction-from-depth/

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
            represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> normals_from_depth_improved(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    KORNIA_CHECK_SHAPE(depth, ["B", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

    # compute the 3d points from depth
    p0: Tensor = depth_to_3d(depth, camera_matrix, normalize_points)  # Bx3xHxW

    p0_padded: Tensor = F.pad(p0, [1, 1, 1, 1], mode="constant")

    h_l, h_r = p0_padded[:, :, 1:-1, :-2], p0_padded[:, :, 1:-1, 2:]  # right and left
    v_u, v_d = p0_padded[:, :, :-2, 1:-1], p0_padded[:, :, 2:, 1:-1]  # up and down

    z: Tensor = p0[:, 2:3, :, :]
    # create masks:
    #  mask_r - horizontal right point is closer than left point
    #  mask_l - horizontal left point is closer than right point
    #  mask_d - vertical down point is closer than up point
    #  mask_u - vertical up point is closer than down point
    mask_r: Tensor = (h_r[:, 2:3, :, :] - z).abs() < (h_l[:, 2:3, :, :] - z).abs()
    mask_l: Tensor = ~mask_r
    mask_d: Tensor = (v_d[:, 2:3, :, :] - z).abs() < (v_u[:, 2:3, :, :] - z).abs()
    mask_u: Tensor = ~mask_d

    # create masks:
    #  mask_rd - best triangle is right and down
    #  mask_ru - best triangle is right and up
    #  mask_ld - best triangle is left and down
    #  mask_lu - best triangle is left and up
    mask_rd: Tensor = (mask_r & mask_d).expand_as(p0)
    mask_ru: Tensor = (mask_r & mask_u).expand_as(p0)
    mask_ld: Tensor = (mask_l & mask_d).expand_as(p0)
    mask_lu: Tensor = (mask_l & mask_u).expand_as(p0)

    p1: Tensor = torch.zeros_like(p0)
    p1[mask_rd] = v_d[mask_rd]
    p1[mask_ru] = h_r[mask_ru]
    p1[mask_ld] = h_l[mask_ld]
    p1[mask_lu] = v_u[mask_lu]

    p2: Tensor = torch.zeros_like(p0)
    p2[mask_rd] = h_r[mask_rd]
    p2[mask_ru] = v_u[mask_ru]
    p2[mask_ld] = v_d[mask_ld]
    p2[mask_lu] = h_l[mask_lu]

    normals: Tensor = torch.cross(p2 - p0, p1 - p0, dim=1)
    return F.normalize(normals, dim=1, p=2)


def normals_from_depth_least_squares(
    depth: Tensor,
    camera_matrix: Tensor,
    *,
    normalize_points: bool = False,
    k: int = 3,
    weighted: bool = True,
    temperature: float = 1.0,
    epsilon: float = 1e-9,
) -> Tensor:
    r"""Compute the normal surface per pixel.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
            represented as the Euclidean ray length from the camera position.
        k: size of local neighborhood to regress for the normal surface per pixel.
        weighted: whether to use the whether least squares.
        temperature: temperature of exponential distance.
        epsilon: for numerical stability.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> normals_from_depth_least_squares(depth, K).shape
        torch.Size([1, 3, 4, 4])

    Notes:
        The normal in the i'th pixel satisfies the over determined linear system: A n = b, s.t ||n||_2^2 = 1.
        A is the matrix of XYZ coordinates in the 3e points of the K neighboring pixels (with shape (K**2, 3)).
        The solution has the closed form: ((A^T A)^-1 A^T Y) / ||((A^T A)^-1 A^T Y)||_2, where Y is a vector on
        1's with shape (K**2, 1).
        Far pixels may not be on the same tangent plane (e.g edge) and should be filtered out. However, this may
        lead to an uneven number of equations per pixel, hence this method allows the Weighted Least Squares,
        weighting equations by the pixels' distances from the center pixel. For that purpose it used exponential
        distance, such that W_ij = W_ji = exp(-temperature * |z_j - z_j|).
    """
    KORNIA_CHECK_SHAPE(depth, ["B", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

    # compute the 3d points from depth
    xyz: Tensor = depth_to_3d(depth, camera_matrix, normalize_points)  # Bx3xHxW

    pad: int = k // 2

    def pad_unfold_permute(x: Tensor) -> Tensor:
        """Extract sliding windows with shape (k, k).

        Args:
            x: Tensor with shape (B, C, H, W).

        Returns:
            Unfolded tensor with shape (B, C, H, W, k ** 2)
        """
        x_p: Tensor = F.pad(x, [pad, pad, pad, pad], mode="replicate")
        x_p_unfolded: Tensor = x_p.unfold(2, k, 1).unfold(3, k, 1)  # (B, C, H, W, k, k)
        x_p_unfolded: Tensor = x_p_unfolded.reshape(*x_p_unfolded.shape[:-2], -1)  # (B, C, H, W, k ** 2)
        x_p_unfolded: Tensor = x_p_unfolded.permute(0, 2, 3, 4, 1)  # (B, C, H, W, k ** 2, C)
        return x_p_unfolded

    a: Tensor = pad_unfold_permute(xyz)  # (B, H, W, k ** 2, 3)

    if weighted:  # calculate the weights
        px: int = (k**2) // 2
        euclidean_dist: Tensor = torch.norm(a[..., px : px + 1, :] - a[..., :], dim=-1)  # (B, H, W, k ** 2)
        exp_dist: Tensor = torch.exp(-temperature * euclidean_dist)  # (B, H, W, k ** 2)
        w = torch.diag_embed(exp_dist, offset=0, dim1=-2, dim2=-1)  # (B, H, W, k ** 2, k ** 2)
    else:
        w = None  # unused

    def matmul_last_2dims(x: Tensor, y: Tensor) -> Tensor:
        """Matrix multiplication on last two dims of given tensors, where x is with shape (*, i, k)
        and y is with shape (*, k, j). * represents any equal number of first dims."""
        return torch.einsum("...ik,...kj->...ij", x, y)

    # apply the (weighted) least squares
    a_T: Tensor = torch.transpose(a, dim0=-2, dim1=-1)  # (B, H, W, 3, k ** 2)
    y: Tensor = torch.ones(*a.shape[:-1], 1).to(a)  # (B, H, W, k ** 2, 1)

    if weighted:
        w__a = matmul_last_2dims(w, a)  # (B, H, W, k ** 2, 3)
    else:
        w__a = a

    a_T__w__a: Tensor = matmul_last_2dims(a_T, w__a)  # (B, H, W, 3, 3)
    if weighted:  # adding epsilon so matrix will be invertible
        e = torch.eye(3).view(1, 1, 1, 3, 3).to(a_T__w__a)
        a_T__w__a = a_T__w__a + epsilon * e

    if weighted:
        w__y = matmul_last_2dims(w, y)  # (B, H, W, k ** 2, 1)
    else:
        w__y = y

    a_T__w__y: Tensor = matmul_last_2dims(a_T, w__y)  # (B, H, W, 3, 1)

    normals: Tensor = matmul_last_2dims(torch.linalg.inv(a_T__w__a), a_T__w__y).squeeze(-1)  # (B, H, W, 3)
    normals: Tensor = normals.permute(0, 3, 1, 2)  # (B, 3, H, W)

    return F.normalize(normals, dim=1, p=2)
