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
import torch
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

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


#TODO: Temporary shape check functions until KORNIA_CHECK_SHAPE is ready

def check_so2_z_shape(z: torch.Tensor) -> None:
    # Check if input is a scalar/unbatched: shape []
    is_scalar = len(z.shape) == 0
    
    # Check if input is a flat vector: shape [B]
    is_flat = KORNIA_CHECK_SHAPE(z, ["B"], raises=False)
    
    # Check if input is a column vector: shape [B, 1]
    is_column = KORNIA_CHECK_SHAPE(z, ["B", "1"], raises=False)

    if not (is_scalar or is_flat or is_column):
        raise ValueError(f"Invalid input size, we expect [], [B], or [B, 1]. Got: {z.shape}")

def check_so2_t_shape(t: torch.Tensor) -> None:
    # Check if input is a batch of 2D vectors: shape [B, 2]
    is_batch_shape = KORNIA_CHECK_SHAPE(t, ["B", "2"], raises=False)
    
    # Check if input is a single 2D vector: shape [2]
    is_single_shape = KORNIA_CHECK_SHAPE(t, ["2"], raises=False)

    if not (is_batch_shape or is_single_shape):
         raise ValueError(f"Invalid translation shape, we expect [B, 2], or [2] Got: {t.shape}")

def check_so2_theta_shape(theta: torch.Tensor) -> None:
    # Allow scalar
    is_scalar = len(theta.shape) == 0
    
    # Check if input is a flat vector: shape [B]
    is_flat = KORNIA_CHECK_SHAPE(theta, ["B"], raises=False)
    
    # Check if input is a column vector: shape [B, 1]
    is_column = KORNIA_CHECK_SHAPE(theta, ["B", "1"], raises=False)

    if not (is_scalar or is_flat or is_column):
        raise ValueError(f"Invalid input size, we expect [], [B], or [B, 1]. Got: {theta.shape}")


def check_so2_matrix_shape(matrix: torch.Tensor) -> None:
    # Allow a stack of 2x2 matrices [B, 2, 2] OR a single 2x2 matrix [2, 2]
    is_batch = KORNIA_CHECK_SHAPE(matrix, ["B", "2", "2"], raises=False)
    is_single = KORNIA_CHECK_SHAPE(matrix, ["2", "2"], raises=False)
    
    if not (is_batch or is_single):
        raise ValueError(f"Invalid input size, we expect [B, 2, 2] or [2, 2]. Got: {matrix.shape}")


def check_so2_matrix(matrix: torch.Tensor) -> None:
    KORNIA_CHECK_IS_TENSOR(matrix)
    
    # Existing shape validation
    if len(matrix.shape) < 2 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Input size must be (*, 2, 2). Got {matrix.shape}")

    # Check the diagonal: m00 == m11
    # Check the off-diagonal: m01 == -m10
    mask_diag = torch.allclose(matrix[..., 0, 0], matrix[..., 1, 1])
    mask_off_diag = torch.allclose(matrix[..., 0, 1], -matrix[..., 1, 0])

    if not (mask_diag and mask_off_diag):
        raise ValueError("Invalid SO2 rotation matrix: constraints m00==m11 and m01==-m10 not met.")

def check_v_shape(v: torch.Tensor) -> None:
    # Allow a batch of 3D vectors [B, 3] OR a single 3D vector [3]
    is_batch = KORNIA_CHECK_SHAPE(v, ["B", "3"], raises=False)
    is_single = KORNIA_CHECK_SHAPE(v, ["3"], raises=False)

    if not (is_batch or is_single):
        raise ValueError(f"Invalid input shape, we expect [B, 3], [3] Got: {v.shape}")


def check_se2_omega_shape(matrix: torch.Tensor) -> None:
    # Allow a stack of 3x3 matrices [B, 3, 3] OR a single 3x3 matrix [3, 3]
    is_batch = KORNIA_CHECK_SHAPE(matrix, ["B", "3", "3"], raises=False)
    is_single = KORNIA_CHECK_SHAPE(matrix, ["3", "3"], raises=False)

    if not (is_batch or is_single):
        raise ValueError(f"Invalid input size, we expect [B, 3, 3] or [3, 3]. Got: {matrix.shape}")


def check_se2_t_shape(t: torch.Tensor) -> None:
    check_so2_t_shape(t)