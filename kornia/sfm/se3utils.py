"""
Operations over the Lie Group SE(3), for rigid-body transformations in 3D
"""

from typing import Optional

import torch


# Threshold to determine if a quantity can be considered 'small'
_eps = 1e-6


def SO3_hat(omega: torch.Tensor) -> torch.Tensor:
    """Implements the hat operator for SO(3), given an input axis-angle 
    vector omega.

    """
    assert torch.is_tensor(omega), 'Input must be of type torch.tensor.'

    omega_hat = torch.zeros(3, 3).type(omega.dtype).to(omega.device)
    omega_hat[0,1] = -omega[2]
    omega_hat[1,0] = omega[2]
    omega_hat[0,2] = omega[1]
    omega_hat[2,0] = -omega[1]
    omega_hat[1,2] = -omega[0]
    omega_hat[2,1] = omega[0]

    return omega_hat


def SE3_hat(xi: torch.Tensor) -> torch.Tensor:
    """Implements the SE(3) hat operator, given a vector of twist 
    (exponential) coordinates.
    """

    assert torch.is_tensor(xi), 'Input must be of type torch.tensor.'

    v = xi[:3]
    omega = xi[3:]
    omega_hat = SO3_hat(omega)

    xi_hat = torch.zeros(4, 4).type(xi.dtype).to(xi.device)
    xi_hat[0:3, 0:3] = omega_hat
    xi_hat[0:3, 3] = v

    return xi_hat


def SO3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Computes the exponential map for the coordinate-vector omega.
    Returns a 3 x 3 SO(3) matrix.

    """

    assert torch.is_tensor(omega), 'Input must be of type torch.Tensor.'
    
    omega_hat = SO3_hat(omega)

    if omega.norm() < _eps:
        R = torch.eye(3, 3).type(omega.dtype).to(omega.device) + omega_hat
    else:
        theta = omega.norm()
        s = theta.sin()
        c = theta.cos()
        omega_hat_sq = omega_hat.mm(omega_hat)
        # Coefficients of the Rodrigues formula
        A = s / theta
        B = (1 - c) / torch.pow(theta, 2)
        C = (theta - s) / torch.pow(theta, 3)
        R = torch.eye(3, 3).type(omega.dtype).to(omega.device) + A * omega_hat + B * omega_hat_sq

    return R


def SE3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Computes the exponential map for the coordinate-vector xi.
    Returns a 4 x 4 SE(3) matrix.

    """

    assert torch.is_tensor(xi), 'Input must be of type torch.tensor.'
    
    v = xi[:3]
    omega = xi[3:]
    omega_hat = SO3_hat(omega)

    if omega.norm() < _eps:
        R = torch.eye(3, 3).type(omega.dtype).to(omega.device) + omega_hat
        V = torch.eye(3, 3).type(omega.dtype).to(omega.device) + omega_hat
    else:
        theta = omega.norm()
        s = theta.sin()
        c = theta.cos()
        omega_hat_sq = omega_hat.mm(omega_hat)
        # Coefficients of the Rodrigues formula
        A = s / theta
        B = (1 - c) / torch.pow(theta, 2)
        C = (theta - s) / torch.pow(theta, 3)
        R = torch.eye(3, 3).type(omega.dtype).to(omega.device) + A * omega_hat + B * omega_hat_sq
        V = torch.eye(3, 3).type(omega.dtype).to(omega.device) + B * omega_hat + C * omega_hat_sq

    t = torch.mm(V, v.view(3,1))
    last_row = torch.tensor([0,0,0,1]).type(omega.dtype).to(omega.device)

    return torch.cat((torch.cat((R, t), dim=1), last_row.unsqueeze(0)), dim=0)
