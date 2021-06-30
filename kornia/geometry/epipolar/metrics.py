"""Module including useful metrics for Structure from Motion."""

import torch

import kornia


def sampson_epipolar_distance(
    pts1: torch.Tensor, pts2: torch.Tensor, Fm: torch.Tensor, squared: bool = True, eps: float = 1e-8
) -> torch.Tensor:
    r"""Returns Sampson distance for correspondences given the fundamental matrix.

    Args:
        pts1: correspondences from the left images with shape
          (B, N, 2 or 3). If they are not homogeneous, converted automatically.
        pts2: correspondences from the right images with shape
          (B, N, 2 or 3). If they are not homogeneous, converted automatically.
        Fm: Fundamental matrices with shape :math:`(B, 3, 3)`. Called Fm to
          avoid ambiguity with torch.nn.functional.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed Sampson distance with shape :math:`(B, N)`.

    """
    if not isinstance(Fm, torch.Tensor):
        raise TypeError("Fm type is not a torch.Tensor. Got {}".format(type(Fm)))

    if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError("Fm must be a (*, 3, 3) tensor. Got {}".format(Fm.shape))

    if pts1.size(-1) == 2:
        pts1 = kornia.convert_points_to_homogeneous(pts1)

    if pts2.size(-1) == 2:
        pts2 = kornia.convert_points_to_homogeneous(pts2)

    # From Hartley and Zisserman, Sampson error (11.9)
    # sam =  (x'^T F x) ** 2 / (  (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2)) )

    # line1_in_2: torch.Tensor = (F @ pts1.permute(0,2,1)).permute(0,2,1)
    # line2_in_1: torch.Tensor = (F.permute(0,2,1) @ pts2.permute(0,2,1)).permute(0,2,1)
    # Instead we can just transpose F once and switch the order of multiplication
    F_t: torch.Tensor = Fm.permute(0, 2, 1)
    line1_in_2: torch.Tensor = pts1 @ F_t
    line2_in_1: torch.Tensor = pts2 @ Fm

    # numerator = (x'^T F x) ** 2
    numerator: torch.Tensor = (pts2 * line1_in_2).sum(2).pow(2)

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator: torch.Tensor = line1_in_2[..., :2].norm(2, dim=2).pow(2) + line2_in_1[..., :2].norm(2, dim=2).pow(2)
    out: torch.Tensor = numerator / denominator
    if squared:
        return out
    return (out + eps).sqrt()


def symmetrical_epipolar_distance(
    pts1: torch.Tensor, pts2: torch.Tensor, Fm: torch.Tensor, squared: bool = True, eps: float = 1e-8
) -> torch.Tensor:
    r"""Returns symmetrical epipolar distance for correspondences given the fundamental matrix.

    Args:
       pts1: correspondences from the left images with shape
         (B, N, 2 or 3). If they are not homogeneous, converted automatically.
       pts2: correspondences from the right images with shape
         (B, N, 2 or 3). If they are not homogeneous, converted automatically.
       Fm: Fundamental matrices with shape :math:`(B, 3, 3)`. Called Fm to
         avoid ambiguity with torch.nn.functional.
       squared: if True (default), the squared distance is returned.
       eps: Small constant for safe sqrt.

    Returns:
        the computed Symmetrical distance with shape :math:`(B, N)`.

    """
    if not isinstance(Fm, torch.Tensor):
        raise TypeError("Fm type is not a torch.Tensor. Got {}".format(type(Fm)))

    if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError("Fm must be a (*, 3, 3) tensor. Got {}".format(Fm.shape))

    if pts1.size(-1) == 2:
        pts1 = kornia.convert_points_to_homogeneous(pts1)

    if pts2.size(-1) == 2:
        pts2 = kornia.convert_points_to_homogeneous(pts2)

    # From Hartley and Zisserman, symmetric epipolar distance (11.10)
    # sed = (x'^T F x) ** 2 /  (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))

    # line1_in_2: torch.Tensor = (F @ pts1.permute(0,2,1)).permute(0,2,1)
    # line2_in_1: torch.Tensor = (F.permute(0,2,1) @ pts2.permute(0,2,1)).permute(0,2,1)

    # Instead we can just transpose F once and switch the order of multiplication
    F_t: torch.Tensor = Fm.permute(0, 2, 1)
    line1_in_2: torch.Tensor = pts1 @ F_t
    line2_in_1: torch.Tensor = pts2 @ Fm

    # numerator = (x'^T F x) ** 2
    numerator: torch.Tensor = (pts2 * line1_in_2).sum(2).pow(2)

    # denominator_inv =  1/ (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator_inv: torch.Tensor = 1.0 / (line1_in_2[..., :2].norm(2, dim=2).pow(2)) + 1.0 / (
        line2_in_1[..., :2].norm(2, dim=2).pow(2)
    )
    out: torch.Tensor = numerator * denominator_inv
    if squared:
        return out
    return (out + eps).sqrt()
