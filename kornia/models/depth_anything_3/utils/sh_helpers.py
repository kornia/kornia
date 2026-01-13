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

from math import isqrt
import torch
from einops import einsum

try:
    from e3nn.o3 import matrix_to_angles, wigner_D
except ImportError:
    from kornia.models.depth_anything_3.utils.logger import logger

    logger.warn("Dependency 'e3nn' not found. Required for rotating the camera space SH coeff")


def project_to_so3_strict(M: torch.Tensor) -> torch.Tensor:
    if M.shape[-2:] != (3, 3):
        raise ValueError("Input must be a batch of 3x3 matrices (i.e., shape [..., 3, 3]).")

    # 1. Compute SVD
    U, S, Vh = torch.linalg.svd(M)
    V = Vh.mH

    # 2. Handle reflection case (det = -1)
    det_U = torch.det(U)
    det_V = torch.det(V)
    is_reflection = (det_U * det_V) < 0
    correction_sign = torch.where(
        is_reflection[..., None],
        torch.tensor([1, 1, -1.0], device=M.device, dtype=M.dtype),
        torch.tensor([1, 1, 1.0], device=M.device, dtype=M.dtype),
    )
    correction_matrix = torch.diag_embed(correction_sign)
    U_corrected = U @ correction_matrix
    R_so3_initial = U_corrected @ V.transpose(-2, -1)

    # 3. Explicitly ensure determinant is 1 (or extremely close)
    current_det = torch.det(R_so3_initial)
    det_correction_factor = torch.pow(current_det, -1 / 3)[..., None, None]
    R_so3_final = R_so3_initial * det_correction_factor

    return R_so3_final


def rotate_sh(
    sh_coefficients: torch.Tensor,  # "*#batch n"
    rotations: torch.Tensor,  # "*#batch 3 3"
) -> torch.Tensor:  # "*batch n"
    # https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2452412653
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape

    with torch.autocast(device_type=rotations.device.type, enabled=False):
        rotations_float32 = rotations.to(torch.float32)

        # switch axes: yzx -> xyz
        P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).unsqueeze(0).to(rotations_float32)
        permuted_rotations = torch.linalg.inv(P) @ rotations_float32 @ P

        # ensure rotation has det == 1 in float32 type
        permuted_rotations_so3 = project_to_so3_strict(permuted_rotations)

        alpha, beta, gamma = matrix_to_angles(permuted_rotations_so3)
        result = []
        for degree in range(isqrt(n)):
            with torch.device(device):
                sh_rotations = wigner_D(degree, alpha, -beta, gamma).type(dtype)
            sh_rotated = einsum(
                sh_rotations,
                sh_coefficients[..., degree**2 : (degree + 1) ** 2],
                "... i j, ... j -> ... i",
            )
            result.append(sh_rotated)

    return torch.cat(result, dim=-1)
