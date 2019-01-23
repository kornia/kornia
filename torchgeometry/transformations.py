from typing import Optional

import torch


__all__ = [
    "relative_pose",
]


def relative_pose(pose_1: torch.Tensor, pose_2: torch.Tensor, eps: Optional[float] = 1e-6) -> torch.Tensor:
    """Function that computes the relative transformation from a reference
    pose 1 to 2.
    """
    if not torch.is_tensor(pose_1):
        raise TypeError("Input pose_1 type is not a torch.Tensor. Got {}"
                        .format(type(pose_1)))
    if not torch.is_tensor(pose_2):
        raise TypeError("Input pose_2 type is not a torch.Tensor. Got {}"
                        .format(type(pose_2)))
    if not (len(pose_1.shape) == 3 and pose_1.shape[-2:] == (4, 4)):
        raise ValueError("Input must be a of the shape Nx4x4."
                         " Got {}".format(pose_1.shape, pose_2.shape))
    if not pose_1.shape == pose_2.shape:
        raise ValueError("Input pose_1 and pose_2 must be a of the same shape."
                         " Got {}".format(pose_1.shape, pose_2.shape))
    # unpack input data
    rot_mat_1 = pose_1[..., :3, :3]  # Nx3x3
    rot_mat_2 = pose_2[..., :3, :3]  # Nx3x3
    t_vec_1 = pose_1[..., :3, -1:]  # Nx3x1
    t_vec_2 = pose_2[..., :3, -1:]  # Nx3x1

    # compute relative pose
    rot_mat_21 = torch.matmul(rot_mat_2, rot_mat_1.transpose(1, 2))
    t_vec_21 = t_vec_2 - torch.matmul(rot_mat_21, t_vec_1)

    # pack output data
    pose_21 = torch.zeros_like(pose_1)
    pose_21[..., :3, :3] = rot_mat_21
    pose_21[..., :3, -1:] = t_vec_21
    pose_21[..., -1, -1] += 1.0
    return pose_21 + eps


