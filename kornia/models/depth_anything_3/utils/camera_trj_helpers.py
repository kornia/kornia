# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    from kornia.models.depth_anything_3.utils.logger import logger

    logger.warn("Dependency 'scipy' not found. Required for interpolating camera trajectory.")

from kornia.models.depth_anything_3.utils.geometry import as_homogeneous


@torch.no_grad()
def render_stabilization_path(poses, k_size=45):
    """Rendering stabilized camera path.
    poses: [batch, 4, 4] or [batch, 3, 4],

    Return:
        smooth path: [batch 4 4]
    """
    num_frames = poses.shape[0]
    device = poses.device
    dtype = poses.dtype

    # Early exit for trivial cases
    if num_frames <= 1:
        return as_homogeneous(poses)

    # Make k_size safe: positive odd and not larger than num_frames
    # 1) Ensure odd
    k_size = max(k_size, 1)
    if k_size % 2 == 0:
        k_size += 1
    # 2) Cap to num_frames (keep odd)
    max_odd = num_frames if (num_frames % 2 == 1) else (num_frames - 1)
    max_odd = max(max_odd, 1)  # covers num_frames == 0 theoretically
    k_size = min(k_size, max_odd)
    # 3) enforce a minimum of 3 when possible (for better smoothing)
    if num_frames >= 3 and k_size < 3:
        k_size = 3

    input_poses = []
    for i in range(num_frames):
        input_poses.append(torch.cat([poses[i, :3, 0:1], poses[i, :3, 1:2], poses[i, :3, 3:4]], dim=-1))
    input_poses = torch.stack(input_poses)  # (num_frames, 3, 3)

    # Prepare Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(ksize=k_size, sigma=-1).astype(np.float32).squeeze()
    gaussian_kernel = torch.tensor(gaussian_kernel, dtype=dtype, device=device).view(1, 1, -1)
    pad = k_size // 2

    output_vectors = []
    for idx in range(3):  # For r1, r2, t
        vec = input_poses[:, :, idx].T.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, num_frames) -> (1, 1, 3, num_frames)
        # But actually, we want (batch=3, channel=1, width=num_frames)
        # So:
        vec = input_poses[:, :, idx].T.unsqueeze(1)  # (3, 1, num_frames)
        vec_padded = F.pad(vec, (pad, pad), mode="reflect")
        filtered = F.conv1d(vec_padded, gaussian_kernel)
        output_vectors.append(filtered.squeeze(1).T)  # (num_frames, 3)

    output_r1, output_r2, output_t = output_vectors  # Each is (num_frames, 3)

    # Normalize r1 and r2
    output_r1 = output_r1 / output_r1.norm(dim=-1, keepdim=True)
    output_r2 = output_r2 / output_r2.norm(dim=-1, keepdim=True)

    output_poses = []
    for i in range(num_frames):
        output_r3 = torch.linalg.cross(output_r1[i], output_r2[i])
        render_pose = torch.cat(
            [
                output_r1[i].unsqueeze(-1),
                output_r2[i].unsqueeze(-1),
                output_r3.unsqueeze(-1),
                output_t[i].unsqueeze(-1),
            ],
            dim=-1,
        )
        output_poses.append(render_pose[:3, :])
    output_poses = as_homogeneous(torch.stack(output_poses, dim=0))

    return output_poses


@torch.no_grad()
def render_wander_path(
    cam2world: torch.Tensor,
    intrinsic: torch.Tensor,
    h: int,
    w: int,
    num_frames: int = 120,
    max_disp: float = 48.0,
):
    device, dtype = cam2world.device, cam2world.dtype
    fx = intrinsic[0, 0] * w
    r = max_disp / fx
    th = torch.linspace(0, 2.0 * torch.pi, steps=num_frames, device=device, dtype=dtype)
    x = r * torch.sin(th)
    yz = r * torch.cos(th) / 3.0
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(num_frames, 1, 1)
    T[:, :3, 3] = torch.stack([x, yz, yz], dim=-1) * -1.0
    c2ws = cam2world.unsqueeze(0) @ T
    # Start at reference pose and end back at reference pose
    c2ws = torch.cat([cam2world.unsqueeze(0), c2ws, cam2world.unsqueeze(0)], dim=0)
    Ks = intrinsic.unsqueeze(0).repeat(c2ws.shape[0], 1, 1)
    return c2ws, Ks


@torch.no_grad()
def render_dolly_zoom_path(
    cam2world: torch.Tensor,
    intrinsic: torch.Tensor,
    h: int,
    w: int,
    num_frames: int = 120,
    max_disp: float = 0.1,
    D_focus: float = 10.0,
):
    device, dtype = cam2world.device, cam2world.dtype
    fx0, fy0 = intrinsic[0, 0] * w, intrinsic[1, 1] * h
    t = torch.linspace(0.0, 2.0, steps=num_frames, device=device, dtype=dtype)
    z = 0.5 * (1.0 - torch.cos(torch.pi * t)) * max_disp
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(num_frames, 1, 1)
    T[:, 2, 3] = -z
    c2ws = cam2world.unsqueeze(0) @ T
    Df = torch.as_tensor(D_focus, device=device, dtype=dtype)
    scale = (Df / (Df + z)).clamp(min=1e-6)
    Ks = intrinsic.unsqueeze(0).repeat(num_frames, 1, 1)
    Ks[:, 0, 0] = (fx0 * scale) / w
    Ks[:, 1, 1] = (fy0 * scale) / h
    return c2ws, Ks


@torch.no_grad()
def interpolate_intrinsics(
    initial: torch.Tensor,  # "*#batch 3 3"
    final: torch.Tensor,  # "*#batch 3 3"
    t: torch.Tensor,  # " time_step"
) -> torch.Tensor:  # "*batch time_step 3 3"
    initial = rearrange(initial, "... i j -> ... () i j")
    final = rearrange(final, "... i j -> ... () i j")
    t = rearrange(t, "t -> t () ()")
    return initial + (final - initial) * t


def intersect_rays(
    a_origins: torch.Tensor,  # "*#batch dim"
    a_directions: torch.Tensor,  # "*#batch dim"
    b_origins: torch.Tensor,  # "*#batch dim"
    b_directions: torch.Tensor,  # "*#batch dim"
) -> torch.Tensor:  # "*batch dim"
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """
    # Broadcast and stack the tensors.
    a_origins, a_directions, b_origins, b_directions = torch.broadcast_tensors(
        a_origins, a_directions, b_origins, b_directions
    )
    origins = torch.stack((a_origins, b_origins), dim=-2)
    directions = torch.stack((a_directions, b_directions), dim=-2)

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "... n i, ... n j -> ... n i j")
    n = n - torch.eye(3, dtype=origins.dtype, device=origins.device)

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "... n i j -> ... i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "... n i j, ... n j -> ... n i")
    rhs = reduce(rhs, "... n i -> ... i", "sum")

    # Left-matrix-multiply both sides by the inverse of lhs to find p.
    return torch.linalg.lstsq(lhs, rhs).solution


def normalize(a: torch.Tensor) -> torch.Tensor:  # "*#batch dim" -> "*#batch dim"
    return a / a.norm(dim=-1, keepdim=True)


def generate_coordinate_frame(
    y: torch.Tensor,  # "*#batch 3"
    z: torch.Tensor,  # "*#batch 3"
) -> torch.Tensor:  # "*batch 3 3"
    """Generate a coordinate frame given perpendicular, unit-length Y and Z vectors."""
    y, z = torch.broadcast_tensors(y, z)
    return torch.stack([y.cross(z, dim=-1), y, z], dim=-1)


def generate_rotation_coordinate_frame(
    a: torch.Tensor,  # "*#batch 3"
    b: torch.Tensor,  # "*#batch 3"
    eps: float = 1e-4,
) -> torch.Tensor:  # "*batch 3 3"
    """Generate a coordinate frame where the Y direction is normal to the plane defined
    by unit vectors a and b. The other axes are arbitrary.
    """
    device = a.device

    # Replace every entry in b that's parallel to the corresponding entry in a with an
    # arbitrary vector.
    b = b.detach().clone()
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 0, 1], dtype=b.dtype, device=device)
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 1, 0], dtype=b.dtype, device=device)

    # Generate the coordinate frame. The initial cross product defines the plane.
    return generate_coordinate_frame(normalize(torch.linalg.cross(a, b)), a)


def matrix_to_euler(
    rotations: torch.Tensor,  # "*batch 3 3"
    pattern: str,
) -> torch.Tensor:  # "*batch 3"
    *batch, _, _ = rotations.shape
    rotations = rotations.reshape(-1, 3, 3)
    angles_np = R.from_matrix(rotations.detach().cpu().numpy()).as_euler(pattern)
    rotations = torch.tensor(angles_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3)


def euler_to_matrix(
    rotations: torch.Tensor,  # "*batch 3"
    pattern: str,
) -> torch.Tensor:  # "*batch 3 3"
    *batch, _ = rotations.shape
    rotations = rotations.reshape(-1, 3)
    matrix_np = R.from_euler(pattern, rotations.detach().cpu().numpy()).as_matrix()
    rotations = torch.tensor(matrix_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3, 3)


def extrinsics_to_pivot_parameters(
    extrinsics: torch.Tensor,  # "*#batch 4 4"
    pivot_coordinate_frame: torch.Tensor,  # "*#batch 3 3"
    pivot_point: torch.Tensor,  # "*#batch 3"
) -> torch.Tensor:  # "*batch 5"
    """Convert the extrinsics to a representation with 5 degrees of freedom:
    1. Distance from pivot point in the "X" (look cross pivot axis) direction.
    2. Distance from pivot point in the "Y" (pivot axis) direction.
    3. Distance from pivot point in the Z (look) direction
    4. Angle in plane
    5. Twist (rotation not in plane)
    """
    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    # Compute the translation elements of the pivot parametrization.
    translation_frame = generate_coordinate_frame(pivot_axis, extrinsics[..., :3, 2])
    origin = extrinsics[..., :3, 3]
    delta = pivot_point - origin
    translation = einsum(translation_frame, delta, "... i j, ... i -> ... j")

    # Add the rotation elements of the pivot parametrization.
    inverted = pivot_coordinate_frame.inverse() @ extrinsics[..., :3, :3]
    y, _, z = matrix_to_euler(inverted, "YXZ").unbind(dim=-1)

    return torch.cat([translation, y[..., None], z[..., None]], dim=-1)


def pivot_parameters_to_extrinsics(
    parameters: torch.Tensor,  # "*#batch 5"
    pivot_coordinate_frame: torch.Tensor,  # "*#batch 3 3"
    pivot_point: torch.Tensor,  # "*#batch 3"
) -> torch.Tensor:  # "*batch 4 4"
    translation, y, z = parameters.split((3, 1, 1), dim=-1)

    euler = torch.cat((y, torch.zeros_like(y), z), dim=-1)
    rotation = pivot_coordinate_frame @ euler_to_matrix(euler, "YXZ")

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    translation_frame = generate_coordinate_frame(pivot_axis, rotation[..., :3, 2])
    delta = einsum(translation_frame, translation, "... i j, ... j -> ... i")
    origin = pivot_point - delta

    *batch, _ = origin.shape
    extrinsics = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    extrinsics = extrinsics.broadcast_to((*batch, 4, 4)).clone()
    extrinsics[..., 3, 3] = 1
    extrinsics[..., :3, :3] = rotation
    extrinsics[..., :3, 3] = origin
    return extrinsics


def interpolate_circular(
    a: torch.Tensor,  # "*#batch"
    b: torch.Tensor,  # "*#batch"
    t: torch.Tensor,  # "*#batch"
) -> torch.Tensor:  # " *batch"
    a, b, t = torch.broadcast_tensors(a, b, t)

    tau = 2 * torch.pi
    a = a % tau
    b = b % tau

    # Consider piecewise edge cases.
    d = (b - a).abs()
    a_left = a - tau
    d_left = (b - a_left).abs()
    a_right = a + tau
    d_right = (b - a_right).abs()
    use_d = (d < d_left) & (d < d_right)
    use_d_left = (d_left < d_right) & (~use_d)
    use_d_right = (~use_d) & (~use_d_left)

    result = a + (b - a) * t
    result[use_d_left] = (a_left + (b - a_left) * t)[use_d_left]
    result[use_d_right] = (a_right + (b - a_right) * t)[use_d_right]

    return result


def interpolate_pivot_parameters(
    initial: torch.Tensor,  # "*#batch 5"
    final: torch.Tensor,  # "*#batch 5"
    t: torch.Tensor,  # " time_step"
) -> torch.Tensor:  # "*batch time_step 5"
    initial = rearrange(initial, "... d -> ... () d")
    final = rearrange(final, "... d -> ... () d")
    t = rearrange(t, "t -> t ()")
    ti, ri = initial.split((3, 2), dim=-1)
    tf, rf = final.split((3, 2), dim=-1)

    t_lerp = ti + (tf - ti) * t
    r_lerp = interpolate_circular(ri, rf, t)

    return torch.cat((t_lerp, r_lerp), dim=-1)


@torch.no_grad()
def interpolate_extrinsics(
    initial: torch.Tensor,  # "*#batch 4 4"
    final: torch.Tensor,  # "*#batch 4 4"
    t: torch.Tensor,  # " time_step"
    eps: float = 1e-4,
) -> torch.Tensor:  # "*batch time_step 4 4"
    """Interpolate extrinsics by rotating around their "focus point," which is the
    least-squares intersection between the look vectors of the initial and final
    extrinsics.
    """
    initial = initial.type(torch.float64)
    final = final.type(torch.float64)
    t = t.type(torch.float64)

    # Based on the dot product between the look vectors, pick from one of two cases:
    # 1. Look vectors are parallel: interpolate about their origins' midpoint.
    # 3. Look vectors aren't parallel: interpolate about their focus point.
    initial_look = initial[..., :3, 2]
    final_look = final[..., :3, 2]
    dot_products = einsum(initial_look, final_look, "... i, ... i -> ...")
    parallel_mask = (dot_products.abs() - 1).abs() < eps

    # Pick focus points.
    initial_origin = initial[..., :3, 3]
    final_origin = final[..., :3, 3]
    pivot_point = 0.5 * (initial_origin + final_origin)
    pivot_point[~parallel_mask] = intersect_rays(
        initial_origin[~parallel_mask],
        initial_look[~parallel_mask],
        final_origin[~parallel_mask],
        final_look[~parallel_mask],
    )

    # Convert to pivot parameters.
    pivot_frame = generate_rotation_coordinate_frame(initial_look, final_look, eps=eps)
    initial_params = extrinsics_to_pivot_parameters(initial, pivot_frame, pivot_point)
    final_params = extrinsics_to_pivot_parameters(final, pivot_frame, pivot_point)

    # Interpolate the pivot parameters.
    interpolated_params = interpolate_pivot_parameters(initial_params, final_params, t)

    # Convert back.
    return pivot_parameters_to_extrinsics(
        interpolated_params.type(torch.float32),
        rearrange(pivot_frame, "... i j -> ... () i j").type(torch.float32),
        rearrange(pivot_point, "... xyz -> ... () xyz").type(torch.float32),
    )


@torch.no_grad()
def generate_wobble_transformation(
    radius: torch.Tensor,  # "*#batch"
    t: torch.Tensor,  # " time_step"
    num_rotations: int = 1,
    scale_radius_with_t: bool = True,
) -> torch.Tensor:  # "*batch time_step 4 4"]:
    # Generate a translation in the image plane.
    tf = torch.eye(4, dtype=torch.float32, device=t.device)
    tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
    radius = radius[..., None]
    if scale_radius_with_t:
        radius = radius * t
    tf[..., 0, 3] = torch.sin(2 * torch.pi * num_rotations * t) * radius
    tf[..., 1, 3] = -torch.cos(2 * torch.pi * num_rotations * t) * radius
    return tf


@torch.no_grad()
def render_wobble_inter_path(cam2world: torch.Tensor, intr_normed: torch.Tensor, inter_len: int, n_skip: int = 3):
    """cam2world: [batch, 4, 4],
    intr_normed: [batch, 3, 3]
    """
    frame_per_round = n_skip * inter_len
    num_rotations = 1

    t = torch.linspace(0, 1, frame_per_round, dtype=torch.float32, device=cam2world.device)
    # t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
    tgt_c2w_b = []
    tgt_intr_b = []
    for b_idx in range(cam2world.shape[0]):
        tgt_c2w = []
        tgt_intr = []
        for cur_idx in range(0, cam2world.shape[1] - n_skip, n_skip):
            origin_a = cam2world[b_idx, cur_idx, :3, 3]
            origin_b = cam2world[b_idx, cur_idx + n_skip, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            if cur_idx == 0:
                delta_prev = delta
            else:
                delta = (delta_prev + delta) / 2
                delta_prev = delta
            tf = generate_wobble_transformation(
                radius=delta * 0.5,
                t=t,
                num_rotations=num_rotations,
                scale_radius_with_t=False,
            )
            cur_extrs = (
                interpolate_extrinsics(
                    cam2world[b_idx, cur_idx],
                    cam2world[b_idx, cur_idx + n_skip],
                    t,
                )
                @ tf
            )
            tgt_c2w.append(cur_extrs[(0 if cur_idx == 0 else 1) :])
            tgt_intr.append(
                interpolate_intrinsics(
                    intr_normed[b_idx, cur_idx],
                    intr_normed[b_idx, cur_idx + n_skip],
                    t,
                )[(0 if cur_idx == 0 else 1) :]
            )
        tgt_c2w_b.append(torch.cat(tgt_c2w))
        tgt_intr_b.append(torch.cat(tgt_intr))
    tgt_c2w = torch.stack(tgt_c2w_b)  # b v 4 4
    tgt_intr = torch.stack(tgt_intr_b)  # b v 3 3
    return tgt_c2w, tgt_intr
