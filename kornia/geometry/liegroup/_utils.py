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

from kornia.core import Tensor


# TODO: Temporary shape check functions until KORNIA_CHECK_SHAPE is ready
def check_so2_z_shape(z: Tensor) -> None:
    z_shape = z.shape
    len_z_shape = len(z_shape)
    if (len_z_shape == 2 and z_shape[1] != 1) or (len_z_shape == 0 and not z.numel()) or (len_z_shape > 2):
        raise ValueError(f"Invalid input size, we expect [B]. Got: {z.shape}")


def check_so2_t_shape(t: Tensor) -> None:
    t_shape = t.shape
    len_t_shape = len(t_shape)
    if ((len_t_shape == 2) and (t_shape[1] != 2)) or ((len_t_shape == 1) and (t_shape[0] != 2)) or (len_t_shape > 2):
        raise ValueError(f"Invalid translation shape, we expect [B, 2], or [2] Got: {t_shape}")


def check_so2_theta_shape(theta: Tensor) -> None:
    theta_shape = theta.shape
    len_theta_shape = len(theta_shape)
    if (
        (len_theta_shape == 2 and theta_shape[1] != 1)
        or (len_theta_shape == 0 and not theta.numel())
        or (len_theta_shape > 2)
    ):
        raise ValueError(f"Invalid input size, we expect [B]. Got: {theta_shape}")


def check_so2_matrix_shape(matrix: Tensor) -> None:
    matrix_shape = matrix.shape
    len_matrix_shape = len(matrix_shape)
    if (
        (len_matrix_shape == 3 and (matrix_shape[1] != 2 or matrix_shape[2] != 2))
        or (len_matrix_shape == 2 and (matrix_shape[0] != 2 or matrix_shape[1] != 2))
        or (len_matrix_shape > 3 or len_matrix_shape < 2)
    ):
        raise ValueError(f"Invalid input size, we expect [B, 2, 2] or [2, 2]. Got: {matrix_shape}")


def check_so2_matrix(matrix: Tensor) -> None:
    for m in matrix.reshape(-1, 2, 2):
        if m[0, 0] != m[1, 1] or m[0, 1] != -m[1, 0]:
            raise ValueError("Invalid rotation matrix")


def check_se2_t_shape(t: Tensor) -> None:
    check_so2_t_shape(t)


def check_v_shape(v: Tensor) -> None:
    v_shape = v.shape
    len_v_shape = len(v_shape)
    if ((len_v_shape == 2) and (v_shape[1] != 3)) or ((len_v_shape == 1) and (v_shape[0] != 3)) or (len_v_shape > 3):
        raise ValueError(f"Invalid input shape, we expect [B, 3], [3] Got: {v_shape}")


def check_se2_omega_shape(matrix: Tensor) -> None:
    matrix_shape = matrix.shape
    len_matrix_shape = len(matrix_shape)
    if (
        (len_matrix_shape == 3 and (matrix_shape[1] != 3 or matrix_shape[2] != 3))
        or (len_matrix_shape == 2 and (matrix_shape[0] != 3 or matrix_shape[1] != 3))
        or (len_matrix_shape > 3 or len_matrix_shape < 2)
    ):
        raise ValueError(f"Invalid input size, we expect [B, 3, 3] or [3, 3]. Got: {matrix_shape}")
