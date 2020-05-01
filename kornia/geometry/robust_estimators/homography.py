import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import kornia

TupleTensor = Tuple[torch.Tensor, torch.Tensor]


def normalize_points(points: torch.Tensor, eps: float = 1e-6) -> TupleTensor:
    '''Function, which normalizes points to have zero mean and unit variance
    and returns corresponding transform
    See "In Defence of the 8-point Algorithm"
    http://www.cs.cmu.edu/afs/andrew/scs/cs/15-463/f07/proj_final/www/amichals/fundamental.pdf
    '''
    assert len(points.shape) == 3, points.shape
    assert points.shape[-1] == 2, points.shape

    x_mean = torch.mean(points, dim=-2, keepdim=True)  # Bx1x2

    x_dist = torch.norm(points - x_mean, dim=-1).unsqueeze(-1)  # BxNx1
    x_dist = torch.mean(x_dist, dim=-2, keepdim=True)  # Bx1x1

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


def find_homography_dlt(points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    '''Function, which finds homography via weighted least squares'''
    assert points1.shape == points2.shape, points1.shape
    assert len(points1.shape) >= 1 and points1.shape[-1] == 2, points1.shape
    assert points1.shape[1] >= 4, points1.shape

    eps: float = 1e-6
    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    # DIAPO 11: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf  # noqa: E501
    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2], dim=-1)
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1, -x2], dim=-1)

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
    try:
        U, S, V = torch.svd(A)
    except:
        return torch.empty(points1_norm.size(0), 3, 3)
    H = V[..., -1].view(-1, 3, 3)
    H = transform2.inverse() @ (H @ transform1)
    H_norm = H / (H[..., -1:, -1:] + eps)
    return H_norm


def find_homography_dlt_iterated(points1: torch.Tensor,
                                 points2: torch.Tensor,
                                 weights: torch.Tensor,
                                 n_iter: int = 10) -> torch.Tensor:
    '''Function, which finds homography via iteratively-reweighted
    least squares ToDo: add citation'''
    H = find_homography_dlt(points1, points2, weights)
    for i in range(n_iter):
        pts1_in_2 = kornia.transform_points(H, points1)
        error = (pts1_in_2 - points2).pow(2).sum(dim=-1)
        error_norm = F.normalize(1.0 / (error + 0.1), dim=1, p=1)
        H = find_homography_dlt(points1, points2, error_norm)
    return H
