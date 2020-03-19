from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia


TupleTensor = Tuple[torch.Tensor, torch.Tensor]


def scale_homography(H, shape_in, shape_out):
    s_factor = float(shape_out[-1]) / shape_in[-1]
    S = torch.tensor([[
        [s_factor, 0., 0.],
        [0., s_factor, 0.],
        [0., 0., 1.],
    ]]).to(H.device)
    return S @ (H @ torch.inverse(S))


def scale_intrinsics(K: torch.Tensor, scale_factor: float) -> torch.Tensor:
    K_scale = K.clone()
    K_scale[..., 0, 0].mul_(scale_factor)
    K_scale[..., 1, 1].mul_(scale_factor)
    K_scale[..., 0, 2].mul_(scale_factor)
    K_scale[..., 1, 2].mul_(scale_factor)
    return K_scale


# NOTE: check and same as opencv
def compute_correspond_epilines(points: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    points_h: torch.Tensor = kornia.convert_points_to_homogeneous(points)

    a, b, c = torch.chunk(F @ points_h.permute(0, 2, 1), dim=1, chunks=3)

    nu = a * a + b * b
    nu = torch.where(nu > 0., 1. / torch.sqrt(nu), torch.ones_like(nu))

    line = torch.cat([a * nu, b * nu, c * nu], dim=1)  # Bx3xN
    return line.permute(0, 2, 1)  # BxNx3


# https://github.com/royshil/morethantechnical/blob/master/opencv_ar/vgg_funcs/vgg_multiview/vgg_F_from_P.m
def fundamental_from_projections(P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:

    X1 = P1[:, [1, 2]]
    X2 = P1[:, [2, 0]]
    X3 = P1[:, [0, 1]]
    Y1 = P2[:, [1, 2]]
    Y2 = P2[:, [2, 0]]
    Y3 = P2[:, [0, 1]]

    def det(a, b):
        return torch.det(torch.cat([a, b], dim=1))

    F = torch.stack([
        det(X1, Y1), det(X2, Y1), det(X3, Y1),
        det(X1, Y2), det(X2, Y2), det(X3, Y2),
        det(X1, Y3), det(X2, Y3), det(X3, Y3)
    ], dim=-1).view(-1, 3, 3)
 
    return F


def projection_from_KRt(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    assert len(K.shape) == 3 and K.shape[-2:] == (3, 3), K.shape
    assert len(R.shape) == 3 and R.shape[-2:] == (3, 3), R.shape
    assert len(t.shape) == 2 and t.shape[-1] == (3), t.shape

    Rt: torch.Tensor = torch.cat([R, t[..., None]], dim=-1)  # Bx3x4
    Rt_h = F.pad(Rt, (0, 0, 0, 1), "constant", 0.)
    Rt_h[..., -1, -1] += 1.

    K_h: torch.Tensor = F.pad(K, (0, 1, 0, 1), "constant", 0.)
    K_h[..., -1, -1] += 1.

    P: torch.Tensor = K_h @ Rt_h
    return P


def normalize_points(points: torch.Tensor, eps: float = 1e-6) -> TupleTensor:
    assert len(points.shape) == 3, points.shape
    assert points.shape[-1] == 2, points.shape

    x_mean = torch.mean(points, dim=-2, keepdim=True)  # Bx1x2

    x_dist = torch.norm(points - x_mean, dim=-1).unsqueeze(-1) # BxNx1
    x_dist = torch.mean(x_dist, dim=-2, keepdim=True) # Bx1x1

    scale = torch.sqrt(torch.tensor(2.)) / (x_dist + eps)  # Bx1x1
    ones, zeros = torch.ones_like(scale), torch.zeros_like(scale)

    transform = torch.cat([
        scale, zeros, -scale * x_mean[..., 0],
        zeros, scale, -scale * x_mean[..., 1],
        zeros, zeros, ones], dim=-1)  # Bx9

    transform = transform.view(-1, 3, 3)  # Bx3x3
    transform = transform.detach()
    points_norm = kornia.transform_points(transform, points)  # BxNx2

    return points_norm, transform


def normalize_fundamental(F: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert len(F.shape) == 3 and F.shape[-2:] == (3, 3), F.shape
    F_val = F[..., 2, 2]
    F_norm = torch.where(F_val.abs() > eps, F / F_val, F)
    return F_norm


def find_homography(points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    assert points1.shape == points2.shape, points1.shape
    assert len(points1.shape) >= 1 and points1.shape[-1] == 2, points1.shape
    assert points1.shape[1] >= 4, points1.shape

    eps: float = 1e-6
    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    # DIAPO 11: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf
    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2*x1, y2*y1, y2], dim=-1)
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2*x1, -x2*y1, -x2], dim=-1)

    w_list = []
    axy_list = []
    for i in range(points1.shape[1]):
        axy_list.append(ax[:, i])
        axy_list.append(ay[:, i])
        w_list.append(weights[:, i])
        w_list.append(weights[:, i])
    A = torch.stack(axy_list, dim=1)
    w = torch.stack(w_list, dim=1)

    # apply weights
    w_diag = torch.diag_embed(w)
    A = A.transpose(-2, -1) @ w_diag @ A

    U, S, V = torch.svd(A)

    H = V[..., -1].view(-1, 3, 3)
    H = transform2.inverse() @ (H @ transform1)

    H_norm = H / (H[..., -1:, -1:] + eps)

    return H_norm


def find_fundamental(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:

    #import pdb;pdb.set_trace()
    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm[:, :2], dim=1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm[:, :2], dim=1, chunks=2)  # Bx1xN

    ones = torch.ones_like(x1)

    # solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x*x', x*y', x, y*x', y*y', y, x', y', 1]
    #X = torch.cat(
    #    [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones],
    #    dim=1).permute(0, 2, 1)  # BxNx9
    X = torch.cat([
        points1_norm[..., 0:1] * points2_norm,
        points1_norm[..., 1:2] * points2_norm,
        points2_norm
    ], dim=1).permute(0, 2, 1)

    #import pdb;pdb;pdb.set_trace()
    _, _, V = torch.svd(X)
    F = V[..., -1].view(-1, 3, 3)

    # reconstruct and force rank2
    U, S, V = torch.svd(F)

    rank_mask = torch.tensor([1., 1., 0]).to(F.device)
    F_projected = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))

    F_est = transform1.transpose(-2, -1) @ (F_projected @ transform2)
 
    # normalize matrix
    F_est_norm = normalize_fundamental(F_est)

    return F_est_norm


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


def test_edgar_fundamental():
    import cv2
    pts1 = (torch.rand([1, 16, 2]) * 100).round().to(torch.float)
    pts2 = pts1.clone() + torch.tensor([2., 3.]).to(torch.float)

    import pdb;pdb.set_trace()
    F_gt, _ = cv2.findFundamentalMat(pts1.numpy(), pts2.numpy(), cv2.FM_8POINT)
    print(f'\nOpencv F:\n {F_gt}')

    F_est = find_fundamental(pts1, pts2)
    print(f'\nEstimated F:\n {F_est}')


def test_epilines():
    import cv2
    points = torch.rand(1, 2, 2)
    F = torch.rand(1, 3, 3)
    line = compute_correspond_epilines(points, F)
    print(line)

    line_cv = cv2.computeCorrespondEpilines(points.numpy().reshape(-1, 1, 2), 1, F.numpy()[0])
    print(line_cv[:,0])


def test_projections():
    P1 = torch.rand(1, 4, 4)
    P2 = torch.rand_like(P1)
    F = fundamental_from_projections(P1, P2)
    import pdb;pdb.set_trace()

    pass


def test_homography():
    import cv2
    B = 1
    N = 50
    x1 = torch.rand(B, N, 2)
    #x2 = torch.rand(B, N, 2)
    x2 = x1.clone()
    x2[..., -1] += 0.5

    H = find_homography(x1, x2)
    H_cv, _ = cv2.findHomography(x1.numpy(), x2.numpy())
    print(f"Homography Kornia: {H}")
    print(f"Homography OpenCV: {H_cv}")

    # unit test
    from torch.testing import assert_allclose
    X2 = kornia.transform_points(H, x1)
    print(f'Kornia error: {torch.sqrt(torch.sum((X2 - x2)**2,2))}')
    X2 = kornia.transform_points(torch.tensor(H_cv).unsqueeze(0), x1)
    print(f'OpenCV error: {torch.sqrt(torch.sum((X2 - x2)**2,2))}')


if __name__ == "__main__":
   #test_edgar_fundamental()
   #test_edgar_epilines()
   #test_projections()
   test_homography()
