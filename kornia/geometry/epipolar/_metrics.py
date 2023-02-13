"""Module including useful metrics for Structure from Motion."""

from torch import Tensor

from kornia.core.check import KORNIA_CHECK_IS_TENSOR
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.linalg import point_line_distance


def sampson_epipolar_distance(
    pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool = True, eps: float = 1e-8
) -> Tensor:
    """Return Sampson distance for correspondences given the fundamental matrix.

    Args:
        pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed Sampson distance with shape :math:`(*, N)`.
    """
    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")

    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)

    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)

    # From Hartley and Zisserman, Sampson error (11.9)
    # sam =  (x'^T F x) ** 2 / (  (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2)) )

    # line1_in_2 = (F @ pts1.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)
    # line2_in_1 = (F.transpose(dim0=-2, dim1=-1) @ pts2.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)

    # Instead we can just transpose F once and switch the order of multiplication
    F_t: Tensor = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2: Tensor = pts1 @ F_t
    line2_in_1: Tensor = pts2 @ Fm

    # numerator = (x'^T F x) ** 2
    numerator: Tensor = (pts2 * line1_in_2).sum(dim=-1).pow(2)

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator: Tensor = line1_in_2[..., :2].norm(2, dim=-1).pow(2) + line2_in_1[..., :2].norm(2, dim=-1).pow(2)
    out: Tensor = numerator / denominator
    if squared:
        return out
    return (out + eps).sqrt()


def symmetrical_epipolar_distance(
    pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool = True, eps: float = 1e-8
) -> Tensor:
    """Return symmetrical epipolar distance for correspondences given the fundamental matrix.

    Args:
       pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
             converted automatically.
       pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
             converted automatically.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.
       squared: if True (default), the squared distance is returned.
       eps: Small constant for safe sqrt.

    Returns:
        the computed Symmetrical distance with shape :math:`(*, N)`.
    """
    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")

    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)

    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)

    # From Hartley and Zisserman, symmetric epipolar distance (11.10)
    # sed = (x'^T F x) ** 2 /  (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))

    # line1_in_2 = (F @ pts1.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)
    # line2_in_1 = (F.transpose(dim0=-2, dim1=-1) @ pts2.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)

    # Instead we can just transpose F once and switch the order of multiplication
    F_t: Tensor = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2: Tensor = pts1 @ F_t
    line2_in_1: Tensor = pts2 @ Fm

    # numerator = (x'^T F x) ** 2
    numerator: Tensor = (pts2 * line1_in_2).sum(dim=-1).pow(2)

    # denominator_inv =  1/ (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator_inv: Tensor = 1.0 / (line1_in_2[..., :2].norm(2, dim=-1).pow(2)) + 1.0 / (
        line2_in_1[..., :2].norm(2, dim=-1).pow(2)
    )
    out: Tensor = numerator * denominator_inv
    if squared:
        return out
    return (out + eps).sqrt()


def left_to_right_epipolar_distance(pts1: Tensor, pts2: Tensor, Fm: Tensor) -> Tensor:
    r"""Return one-sided epipolar distance for correspondences given the fundamental matrix.

    This method measures the distance from points in the right images to the epilines
    of the corresponding points in the left images as they reflect in the right images.

    Args:
       pts1: correspondences from the left images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       pts2: correspondences from the right images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to
         avoid ambiguity with torch.nn.functional.

    Returns:
        the computed Symmetrical distance with shape :math:`(*, N)`.
    """
    KORNIA_CHECK_IS_TENSOR(pts1)
    KORNIA_CHECK_IS_TENSOR(pts2)
    KORNIA_CHECK_IS_TENSOR(Fm)

    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)

    F_t: Tensor = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2: Tensor = pts1 @ F_t

    return point_line_distance(pts2, line1_in_2)


def right_to_left_epipolar_distance(pts1: Tensor, pts2: Tensor, Fm: Tensor) -> Tensor:
    r"""Return one-sided epipolar distance for correspondences given the fundamental matrix.

    This method measures the distance from points in the left images to the epilines
    of the corresponding points in the right images as they reflect in the left images.

    Args:
       pts1: correspondences from the left images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       pts2: correspondences from the right images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to
         avoid ambiguity with torch.nn.functional.

    Returns:
        the computed Symmetrical distance with shape :math:`(*, N)`.
    """
    KORNIA_CHECK_IS_TENSOR(pts1)
    KORNIA_CHECK_IS_TENSOR(pts2)
    KORNIA_CHECK_IS_TENSOR(Fm)

    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)

    line2_in_1: Tensor = pts2 @ Fm

    return point_line_distance(pts1, line2_in_1)
