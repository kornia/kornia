# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import numpy as np
import torch
from evo.core.trajectory import PosePath3D

from kornia.models.depth_anything_3.utils.geometry import affine_inverse, affine_inverse_np


def batch_apply_alignment_to_enc(
    rots: torch.Tensor, trans: torch.Tensor, scales: torch.Tensor, enc_list: List[torch.Tensor]
):
    pass


def batch_apply_alignment_to_ext(
    rots: torch.Tensor, trans: torch.Tensor, scales: torch.Tensor, ext: torch.Tensor
):
    device, _ = ext.device, ext.dtype
    if ext.shape[-2:] == (3, 4):
        pad = torch.zeros((*ext.shape[:-2], 4, 4), dtype=ext.dtype, device=device)
        pad[..., :3, :4] = ext
        pad[..., 3, 3] = 1.0
        ext = pad
    pose_est = affine_inverse(ext)
    pose_new_align_rot = rots[:, None] @ pose_est[..., :3, :3]
    pose_new_align_trans = (
        scales[:, None, None] * (rots[:, None] @ pose_est[..., :3, 3:])[..., 0] + trans[:, None]
    )
    pose_new_align = torch.zeros_like(ext)
    pose_new_align[..., :3, :3] = pose_new_align_rot
    pose_new_align[..., :3, 3] = pose_new_align_trans
    pose_new_align[..., 3, 3] = 1.0
    return affine_inverse(pose_new_align)[:, :3]


def batch_align_poses_umeyama(ext_ref: torch.Tensor, ext_est: torch.Tensor):
    device, dtype = ext_ref.device, ext_ref.dtype
    assert ext_ref.dtype in [torch.float32, torch.float64]
    assert ext_est.dtype in [torch.float32, torch.float64]
    assert ext_ref.requires_grad is False
    assert ext_est.requires_grad is False
    rots, trans, scales = [], [], []
    for b in range(ext_ref.shape[0]):
        r, t, s = align_poses_umeyama(ext_ref[b].cpu().numpy(), ext_est[b].cpu().numpy())
        rots.append(torch.from_numpy(r).to(device=device, dtype=dtype))
        trans.append(torch.from_numpy(t).to(device=device, dtype=dtype))
        scales.append(torch.tensor(s, device=device, dtype=dtype))
    return torch.stack(rots), torch.stack(trans), torch.stack(scales)


# Dependencies: affine_inverse_np, PosePath3D (maintain consistency with your existing project)


def _to44(ext):
    if ext.shape[1] == 3:
        out = np.eye(4)[None].repeat(len(ext), 0)
        out[:, :3, :4] = ext
        return out
    return ext


def _poses_from_ext(ext_ref, ext_est):
    ext_ref = _to44(ext_ref)
    ext_est = _to44(ext_est)
    pose_ref = affine_inverse_np(ext_ref)
    pose_est = affine_inverse_np(ext_est)
    return pose_ref, pose_est


def _umeyama_sim3_from_paths(pose_ref, pose_est):
    path_ref = PosePath3D(poses_se3=pose_ref.copy())
    path_est = PosePath3D(poses_se3=pose_est.copy())
    r, t, s = path_est.align(path_ref, correct_scale=True)
    pose_est_aligned = np.stack(path_est.poses_se3)
    return r, t, s, pose_est_aligned


def _apply_sim3_to_poses(poses, r, t, s):
    out = poses.copy()
    Ri = poses[:, :3, :3]
    ti = poses[:, :3, 3]
    out[:, :3, :3] = r @ Ri
    out[:, :3, 3] = (r @ (s * ti.T)).T + t
    return out


def _median_nn_thresh(pose_ref, pose_est_aligned):
    P_ref = pose_ref[:, :3, 3]
    P_est = pose_est_aligned[:, :3, 3]
    dists = []
    for p in P_est:
        dd = np.linalg.norm(P_ref - p[None, :], axis=1)
        dists.append(dd.min())
    return float(np.median(dists)) if dists else 0.0


def _ransac_align_sim3(
    pose_ref, pose_est, sub_n=None, inlier_thresh=None, max_iters=10, random_state=None
):
    rng = np.random.default_rng(random_state)
    N = pose_ref.shape[0]
    idx_all = np.arange(N)
    if sub_n is None:
        sub_n = max(3, (N + 1) // 2)
    else:
        sub_n = max(3, min(sub_n, N))

    # Pre-alignment + default threshold
    r0, t0, s0, pose_est0 = _umeyama_sim3_from_paths(pose_ref, pose_est)
    if inlier_thresh is None:
        inlier_thresh = _median_nn_thresh(pose_ref, pose_est0)

    P_ref_all = pose_ref[:, :3, 3]

    best_model = (r0, t0, s0)
    best_inliers = None
    best_score = (-1, np.inf)  # (num_inliers, mean_err)

    for _ in range(max_iters):
        sample = rng.choice(idx_all, size=sub_n, replace=False)
        try:
            r, t, s, _ = _umeyama_sim3_from_paths(pose_ref[sample], pose_est[sample])
        except Exception:
            continue
        pose_h = _apply_sim3_to_poses(pose_est, r, t, s)
        P_h = pose_h[:, :3, 3]
        errs = np.linalg.norm(P_h - P_ref_all, axis=1)  # Match by same index
        inliers = errs <= inlier_thresh
        k = int(inliers.sum())
        mean_err = float(errs[inliers].mean()) if k > 0 else np.inf
        if (k > best_score[0]) or (k == best_score[0] and mean_err < best_score[1]):
            best_score = (k, mean_err)
            best_model = (r, t, s)
            best_inliers = inliers

    # Fit again with best inliers
    if best_inliers is not None and best_inliers.sum() >= 3:
        r, t, s, _ = _umeyama_sim3_from_paths(pose_ref[best_inliers], pose_est[best_inliers])
    else:
        r, t, s = best_model
    return r, t, s


def align_poses_umeyama(
    ext_ref: np.ndarray,
    ext_est: np.ndarray,
    return_aligned=False,
    ransac=False,
    sub_n=None,
    inlier_thresh=None,
    ransac_max_iters=10,
    random_state=None,
):
    """
    Align estimated trajectory to reference using Umeyama Sim(3).
    Default no RANSAC; if ransac=True, use RANSAC (max iterations default 10).
    - sub_n defaults to half the number of frames (rounded up, at least 3)
    - inlier_thresh defaults to median of "distance from each estimated pose to
      nearest reference pose after pre-alignment"
    Returns rotation (3x3), translation (3,), scale; optionally returns aligned extrinsics (4x4).
    """
    pose_ref, pose_est = _poses_from_ext(ext_ref, ext_est)

    if not ransac:
        r, t, s, pose_est_aligned = _umeyama_sim3_from_paths(pose_ref, pose_est)
    else:
        r, t, s = _ransac_align_sim3(
            pose_ref,
            pose_est,
            sub_n=sub_n,
            inlier_thresh=inlier_thresh,
            max_iters=ransac_max_iters,
            random_state=random_state,
        )
        pose_est_aligned = _apply_sim3_to_poses(pose_est, r, t, s)

    if return_aligned:
        ext_est_aligned = affine_inverse_np(pose_est_aligned)
        return r, t, s, ext_est_aligned
    return r, t, s


# def align_poses_umeyama(ext_ref: np.ndarray, ext_est: np.ndarray, return_aligned=False):
#     """
#     Align estimated trajectory to reference trajectory using Umeyama Sim(3)
#     alignment (via evo PosePath3D). # noqa
#     Returns rotation, translation, and scale.
#     """
#     # If input extrinsics are 3x4, convert to 4x4 by padding
#     if ext_ref.shape[1] == 3:
#         ext_ref_ = np.eye(4)[None].repeat(len(ext_ref), 0)
#         ext_ref_[:, :3] = ext_ref
#         ext_ref = ext_ref_
#     if ext_est.shape[1] == 3:
#         ext_est_ = np.eye(4)[None].repeat(len(ext_est), 0)
#         ext_est_[:, :3] = ext_est
#         ext_est = ext_est_

#     # Convert to camera poses (inverse extrinsics)
#     pose_ref = affine_inverse_np(ext_ref)
#     pose_est = affine_inverse_np(ext_est)

#     # Create evo PosePath3D objects
#     path_ref = PosePath3D(poses_se3=pose_ref)
#     path_est = PosePath3D(poses_se3=pose_est)
#     r, t, s = path_est.align(path_ref, correct_scale=True)
#     if return_aligned:
#         return r, t, s, affine_inverse_np(np.stack(path_est.poses_se3))
#     else:
#         return r, t, s


def apply_umeyama_alignment_to_ext(
    rot: np.ndarray,  # (3,3)
    trans: np.ndarray,  # (3,) or (1,3)
    scale: float,
    ext_est: np.ndarray,  # (...,4,4) or (...,3,4)
) -> np.ndarray:
    """
    Apply Sim(3) (R, t, s) to a batch of world-to-camera extrinsics ext_est.
    Returns the aligned extrinsics, with the same shape as input.
    """

    # Allow 3x4 extrinsics: pad to 4x4
    if ext_est.shape[-2:] == (3, 4):
        pad = np.zeros((*ext_est.shape[:-2], 4, 4), dtype=ext_est.dtype)
        pad[..., :3, :4] = ext_est
        pad[..., 3, 3] = 1.0
        ext_est = pad

    # Convert world-to-camera to camera-to-world
    pose_est = affine_inverse_np(ext_est)  # (...,4,4)
    R_e = pose_est[..., :3, :3]  # (...,3,3)
    t_e = pose_est[..., :3, 3]  # (...,3)

    # Apply Sim(3) transformation
    R_a = np.einsum("ij,...jk->...ik", rot, R_e)  # (...,3,3)
    t_a = scale * np.einsum("ij,...j->...i", rot, t_e) + trans  # (...,3)

    # Assemble the transformed pose
    pose_a = np.zeros_like(pose_est)
    pose_a[..., :3, :3] = R_a
    pose_a[..., :3, 3] = t_a
    pose_a[..., 3, 3] = 1.0

    # Convert back to world-to-camera
    return affine_inverse_np(pose_a)


def transform_points_sim3(points, rot, trans, scale, inverse=False):
    """
    Sim(3) transform point cloud
    points: (N, 3)
    rot: (3, 3)
    trans: (3,) or (1, 3)
    scale: float
    inverse: Whether to do inverse transform (ref->est)
    Returns: (N, 3)
    """
    if not inverse:
        # Forward: est -> ref
        return scale * (points @ rot.T) + trans
    else:
        # Inverse: ref -> est
        return ((points - trans) @ rot) / scale


def _rand_rot():
    u1, u2, u3 = np.random.rand(3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.math.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.math.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.math.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.math.pi * u3),
        ]
    )
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rand_pose():
    R, t = _rand_rot(), np.random.randn(3)
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t
    return P


if __name__ == "__main__":
    np.random.seed(42)
    # 1. Randomly generate reference trajectory and Sim(3)
    N = 8
    pose_ref = np.stack([_rand_pose() for _ in range(N)])  # (N,4,4)  cam→world
    rot_gt = _rand_rot()
    scale_gt = 2.3
    trans_gt = np.random.randn(3)
    # 2. Generate estimated trajectory (apply Sim(3))
    pose_est = np.zeros_like(pose_ref)
    for i in range(N):
        R = pose_ref[i][:3, :3]
        t = pose_ref[i][:3, 3]
        pose_est[i][:3, :3] = rot_gt @ R
        pose_est[i][:3, 3] = scale_gt * (rot_gt @ t) + trans_gt
        pose_est[i][3, 3] = 1.0
    # 3. Get extrinsics (world->cam)
    ext_ref = affine_inverse_np(pose_ref)
    ext_est = affine_inverse_np(pose_est)
    # 4. Use umeyama alignment, estimate Sim(3)
    r_est, t_est, s_est = align_poses_umeyama(ext_ref, ext_est)
    print("GT scale:", scale_gt, "Estimated:", s_est)
    print("GT trans:", trans_gt, "Estimated:", t_est)
    print("GT rot:\n", rot_gt, "\nEstimated:\n", r_est)
    # 5. Random point cloud, in ref frame
    num_points = 100
    points_ref = np.random.randn(num_points, 3)
    # 6. Use GT Sim(3) inverse transform to est frame
    points_est = transform_points_sim3(points_ref, rot_gt, trans_gt, scale_gt, inverse=True)
    # 7. Use estimated Sim(3) forward transform back to ref frame
    points_ref_recovered = transform_points_sim3(points_est, r_est, t_est, s_est, inverse=False)
    # 8. Check error
    err = np.abs(points_ref_recovered - points_ref)
    print("Point cloud sim3 transform error (mean abs):", err.mean())
    print("Point cloud sim3 transform error (max abs):", err.max())
    assert err.mean() < 1e-6, "Mean sim3 transform error too large!"
    assert err.max() < 1e-5, "Max sim3 transform error too large!"
    print("Sim(3) point cloud transform & alignment test passed!")
