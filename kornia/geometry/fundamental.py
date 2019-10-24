"""Module containing operators to work with fundamental and essential matrices."""

import torch
from kornia.geometry.conversions import convert_points_to_homogeneous


def raise_error_if_pts_are_not_valid(pts: torch.Tensor, tensor_name: str = 'pts') -> None:
    """Auxilary function, which verifies that input is a torch.tensor of [BxNx2] or [BxNx3] shape

    Args:
        pts (any): input to test
        tensor_name (str): tensor name for error message
    """
    if not isinstance(pts, torch.Tensor):
        raise TypeError("{} type is not a torch.Tensor. Got {}".format(
            tensor_name, type(pts)))

    if (len(pts.shape) != 3) or (pts.size(-1) not in [2, 3]):
        raise ValueError(
            "{} must be a (B, N, 3) or (B, N, 2) tensor. Got {}".format(
                tensor_name, pts.shape))
    return


def symmetrical_epipolar_distance(pts1: torch.Tensor,
                                  pts2: torch.Tensor,
                                  Fm: torch.Tensor,
                                  squared: bool = True,
                                  eps: float = 1e-9) -> torch.Tensor:
    """Returns symmetrical epipolar distance for correspondences given the fundamental matrix

    Arguments:
        pts1 (torch.Tensor): correspondences from the left images. If they are not homogenuous, converted automatically
        pts2 (torch.Tensor): correspondences from the right images.
        Fm (torch.Tensor): fundamental matrices. Called Fm to avoid ambiguity with torch.nn.functional
        squared (bool): if True (default), the squared distance is returned
        eps (float): (default 1e-9) small constant for safe sqrt.

    Shape:
        - Input: :math:`(B, N, 2 or 3)`, :math:`(B, N, 2 or 3)` and :math:`(B, 3, 3)`.
                 Where B - batch size and N - number of correspondences
        - Output: :math:`(B, N)`
    """
    raise_error_if_pts_are_not_valid(pts1, 'pts1')
    raise_error_if_pts_are_not_valid(pts2, 'pts2')

    if not isinstance(Fm, torch.Tensor):
        raise TypeError("Fm type is not a torch.Tensor. Got {}".format(
            type(Fm)))

    if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(
            "Fm must be a (*, 3, 3) tensor. Got {}".format(
                Fm.shape))

    if pts1.size(-1) == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    if pts2.size(-1) == 2:
        pts2 = convert_points_to_homogeneous(pts2)

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
    denominator_inv: torch.Tensor = (1. / (line1_in_2[..., :2].norm(2, dim=2).pow(2)) +
                                     1. / (line2_in_1[..., :2].norm(2, dim=2).pow(2)))
    out: torch.Tensor = numerator * denominator_inv
    if squared:
        return out
    return (out + eps).sqrt()


def sampson_epipolar_distance(pts1: torch.Tensor,
                              pts2: torch.Tensor,
                              Fm: torch.Tensor,
                              squared: bool = True,
                              eps: float = 1e-9) -> torch.Tensor:
    """Returns Sampson distance for correspondences given the fundamental matrix

    Arguments:
        pts1 (torch.Tensor): correspondences from the left images. If they are not homogenuous, converted automatically
        pts2 (torch.Tensor): correspondences from the right images.
        Fm (torch.Tensor): fundamental matrices. Called Fm to avoid ambiguity with torch.nn.functional
        squared (bool): if True (default), the squared distance is returned
        eps (float): (default 1e-9) small constant for safe sqrt.

    Shape:
        - Input: :math:`(B, N, 2 or 3)`, :math:`(B, N, 2 or 3)` and :math:`(B, 3, 3)`. Where B - batch size and
                 N - number of correspondences
        - Output: :math:`(B, N)`
    """
    raise_error_if_pts_are_not_valid(pts1, 'pts1')
    raise_error_if_pts_are_not_valid(pts2, 'pts2')

    if not isinstance(Fm, torch.Tensor):
        raise TypeError("Fm type is not a torch.Tensor. Got {}".format(
            type(Fm)))

    if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(
            "Fm must be a (*, 3, 3) tensor. Got {}".format(
                Fm.shape))

    if pts1.size(-1) == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    if pts2.size(-1) == 2:
        pts2 = convert_points_to_homogeneous(pts2)

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
