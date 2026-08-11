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

import inspect
import sys
import warnings
from functools import partial

import numpy as np
import pytest
import torch

import kornia
from kornia.core._compat import torch_version
from kornia.core.ops import eye_like
from kornia.geometry.conversions import (
    ARKitQTVecs_to_ColmapQTVecs,
    Rt_to_matrix4x4,
    axis_angle_to_rotation_matrix,
    camtoworld_graphics_to_vision_4x4,
    camtoworld_graphics_to_vision_Rt,
    camtoworld_to_worldtocam_Rt,
    camtoworld_vision_to_graphics_4x4,
    camtoworld_vision_to_graphics_Rt,
    euler_from_quaternion,
    matrix4x4_to_Rt,
    quaternion_from_euler,
    worldtocam_to_camtoworld_Rt,
)
from kornia.geometry.quaternion import Quaternion

from testing.base import BaseTester, assert_close


@pytest.fixture()
def atol(device, dtype):
    """Lower tolerance for cuda-float16 only."""
    if "cuda" in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


@pytest.fixture()
def rtol(device, dtype):
    """Lower tolerance for cuda-float16 only."""
    if "cuda" in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


class TestAngleAxisToQuaternion(BaseTester):
    # based on:
    # https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/rotation_test.cc#L271

    def test_smoke(self, device, dtype):
        axis_angle = torch.zeros(3, dtype=dtype, device=device)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.shape == (4,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        axis_angle = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.shape == (batch_size, 4)

    def test_zero_angle(self, device, dtype, atol, rtol):
        axis_angle = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        axis_angle = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        axis_angle = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        axis_angle = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        axis_angle = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, half_sqrt2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        axis_angle = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, half_sqrt2, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        axis_angle = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, 0.0, half_sqrt2), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        axis_angle = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.axis_angle_to_quaternion), (axis_angle,))

    def test_convention_theta_beyond_pi_returns_the_w_negative_half(self, device, dtype):
        # Convention pin: axis_angle_to_quaternion applies w = cos(theta/2) and
        # (x, y, z) = sin(theta/2) * axis verbatim, with NO canonicalisation to w >= 0, so any
        # theta > pi comes back in the w < 0 half of the double cover. A full turn about +x gives
        # (-1, 0, 0, 0) -- the same rotation as the identity quaternion (1, 0, 0, 0) that a
        # canonicalising implementation would return, but not the same four numbers. The second
        # case is off-axis (theta = 4 rad about (1, 2, 3)/sqrt(14)) so that a sign flip on any
        # single component is caught as well.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.cos(math.pi), math.sin(math.pi) -> (-1.0, 1.2246467991473532e-16)
        #   n = math.sqrt(14.0); axis = (1 / n, 2 / n, 3 / n); theta = 4.0
        #   [theta * a for a in axis]
        #     -> [1.0690449676496976, 2.138089935299395, 3.2071349029490928]
        #   [math.cos(theta / 2)] + [math.sin(theta / 2) * a for a in axis]
        #     -> [-0.4161468365471424, 0.24301995956120354, 0.48603991912240707, 0.7290598786836107]
        full_turn = kornia.geometry.conversions.axis_angle_to_quaternion(
            torch.tensor([6.283185307179586, 0.0, 0.0], device=device, dtype=dtype)
        )
        self.assert_close(full_turn, torch.tensor([-1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype))

        off_axis = kornia.geometry.conversions.axis_angle_to_quaternion(
            torch.tensor([1.0690449676496976, 2.138089935299395, 3.2071349029490928], device=device, dtype=dtype)
        )
        self.assert_close(
            off_axis,
            torch.tensor(
                [-0.4161468365471424, 0.24301995956120354, 0.48603991912240707, 0.7290598786836107],
                device=device,
                dtype=dtype,
            ),
        )

    def test_convention_axis_angle_quaternion_roundtrip_is_exact_in_float64(self, device):
        # Convention pin: axis_angle_to_quaternion and quaternion_to_axis_angle are exact inverses
        # in float64 over the whole [0, pi] range, including the two singular points theta = 0 and
        # theta = pi. This is what separates the quaternion leg from the matrix leg: the same
        # round-trip through axis_angle_to_rotation_matrix is only accurate to ~1e-6 even in
        # float64 (see TestRotationMatrixToAngleAxis.test_convention_axis_angle_roundtrip_
        # tolerance_is_1e_6_in_float64), so "the round-trip is exact" is a statement about this
        # pair only. float64 is hardcoded and the dtype fixture dropped because exactness is a
        # float64-only claim; MPS is skipped visibly because it has no float64 at all.
        # Snippet used to generate the inputs (stdlib only):
        #   import math
        #   n = math.sqrt(14.0); axis = (1 / n, 2 / n, 3 / n)
        #   [[theta * a for a in axis] for theta in (0.0, 1e-3, 0.7, math.pi)]
        # Measured max |roundtrip - input| at those four thetas (torch 2.9.1, cpu float64):
        #   0.0, 2.168404344971009e-19, 0.0, 0.0 -- so atol 1e-15 is ~3 orders above the worst
        #   observed error and ~9 orders below the 1e-6 the matrix leg would give.
        if device.type == "mps":
            pytest.skip("MPS has no float64, and this exactness pin is float64-only by construction")

        axis_angle = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0002672612419124244, 0.0005345224838248488, 0.0008017837257372733],
                [0.18708286933869706, 0.3741657386773941, 0.5612486080160912],
                [0.839625954181357, 1.679251908362714, 2.518877862544071],
            ],
            device=device,
            dtype=torch.float64,
        )

        roundtrip = kornia.geometry.conversions.quaternion_to_axis_angle(
            kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        )

        self.assert_close(roundtrip, axis_angle, atol=1e-15, rtol=0.0)

    # Convention pin for the four deprecated aliases (they have no test class and no docstring of
    # their own; this class is named after one of them). Each still works and is a thin wrapper:
    # it emits a DeprecationWarning naming both the old and the new symbol, and returns output
    # that is bit-identical to the replacement's. The replacement, not the alias, is where the
    # Convention block lives.
    # The call is wrapped in warnings.catch_warnings() because invoking a kornia deprecated symbol
    # rewrites the process-global DeprecationWarning filters; pytest.warns alone does not restore
    # them, so without the wrapper this pin would leak filter state into every later test.
    # Snippet used to generate expected (torch only):
    #   import warnings, kornia.geometry.conversions as C
    #   with warnings.catch_warnings(record=True) as w:
    #       warnings.simplefilter("always")
    #       out = C.angle_axis_to_quaternion(torch.tensor([0.1, 0.2, 0.3]))
    #   w[0].category, str(w[0].message) -> DeprecationWarning, 'Since kornia 0.7.0 the
    #     `angle_axis_to_quaternion` is deprecated in favor of `axis_angle_to_quaternion`.'
    #   torch.equal(out, C.axis_angle_to_quaternion(torch.tensor([0.1, 0.2, 0.3]))) -> True
    @pytest.mark.parametrize(
        ("deprecated_name", "replacement_name", "arg"),
        [
            ("angle_axis_to_rotation_matrix", "axis_angle_to_rotation_matrix", [[0.1, 0.2, 0.3]]),
            (
                "rotation_matrix_to_angle_axis",
                "rotation_matrix_to_axis_angle",
                [
                    [0.5357142857142858, -0.6229365034008422, 0.5700529070291328],
                    [0.765793646257985, 0.6428571428571429, -0.01716931065742361],
                    [-0.3557671927434186, 0.4457407392288521, 0.8214285714285714],
                ],
            ),
            ("quaternion_to_angle_axis", "quaternion_to_axis_angle", [1.0, 2.0, 3.0, 4.0]),
            ("angle_axis_to_quaternion", "axis_angle_to_quaternion", [0.1, 0.2, 0.3]),
        ],
    )
    def test_convention_deprecated_alias_warns_and_matches_replacement(
        self, device, dtype, deprecated_name, replacement_name, arg
    ):
        deprecated = getattr(kornia.geometry.conversions, deprecated_name)
        replacement = getattr(kornia.geometry.conversions, replacement_name)
        tensor = torch.tensor(arg, device=device, dtype=dtype)

        expected = replacement(tensor)

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with pytest.warns(
                DeprecationWarning, match=f"`{deprecated_name}` is deprecated in favor of `{replacement_name}`"
            ):
                actual = deprecated(tensor)

        assert torch.equal(actual, expected)


class TestQuaternionToAngleAxis(BaseTester):
    def test_smoke(self, device, dtype):
        quaternion = torch.zeros(4, device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        assert axis_angle.shape == (3,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        assert axis_angle.shape == (batch_size, 3)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi, 0.0, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((np.sqrt(3.0) / 2.0, 0.0, 0.0, 0.5), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 3.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_to_axis_angle), (quaternion,))

    def test_convention_double_cover_q_and_minus_q_give_the_same_axis_angle(self, device, dtype):
        # Convention pin: q and -q are the same rotation (the unit quaternions double-cover SO(3)),
        # and quaternion_to_axis_angle collapses the two onto one bit-identical vector -- it picks
        # the representative with |theta| <= pi rather than propagating the input's sign. torch.equal
        # rather than assert_close because the agreement is exact, not approximate: measured max
        # difference is 0.0 over 500 random float64 quaternions (seeded torch.Generator(6)),
        # bit-identical in 500/500 of them, and at float32/float16/bfloat16 for the pinned input.
        # Pinned on a non-unit, non-axis-aligned quaternion so no symmetry can carry the assertion.
        # Snippet used to generate expected (stdlib only, q = (1, 2, 3, 4) normalised):
        #   import math
        #   u = [v / math.sqrt(30.0) for v in (1.0, 2.0, 3.0, 4.0)]
        #   nv = math.sqrt(u[1] ** 2 + u[2] ** 2 + u[3] ** 2)
        #   theta = 2 * math.atan2(nv, u[0])
        #   [theta * u[i + 1] / nv for i in range(3)]
        #     -> [1.03038058532817, 1.5455708779922552, 2.06076117065634]
        quaternion = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)

        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)

        self.assert_close(
            axis_angle,
            torch.tensor([1.03038058532817, 1.5455708779922552, 2.06076117065634], device=device, dtype=dtype),
        )
        assert torch.equal(kornia.geometry.conversions.quaternion_to_axis_angle(-quaternion), axis_angle)

    def test_convention_quaternion_to_axis_angle_is_scale_invariant(self, device, dtype):
        # Convention pin (quaternion_to_axis_angle has no test class under its own name; this class
        # is the one that exercises it): the function does not require -- and does not check -- a
        # unit quaternion. It is homogeneous in its input, so scaling the whole quaternion leaves
        # the axis-angle vector bit-identical. The scale factors are powers of two so that the
        # scaling itself is exact in binary floating point at every dtype; the invariance is a
        # property of atan2 plus the 2*theta/||v|| factor, not of the particular numbers (verified
        # bit-identical in 500/500 random float64 quaternions, seeded torch.Generator(7), for both
        # factors -- a non-power-of-two factor such as 3 is invariant to rounding only, 82/500).
        # Contrast quaternion_exp_to_log, which does NOT normalise and is silently wrong on a
        # non-unit input.
        quaternion = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)

        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)

        assert torch.equal(kornia.geometry.conversions.quaternion_to_axis_angle(2.0 * quaternion), axis_angle)
        assert torch.equal(kornia.geometry.conversions.quaternion_to_axis_angle(0.5 * quaternion), axis_angle)


class TestRotationMatrixToQuaternion(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix)
        assert quaternion.shape == (batch_size, 4)

    def test_identity(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_rot_x_45(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        pi_half2 = torch.cos(kornia.pi / 4.0).to(device=device, dtype=dtype)
        expected = torch.tensor((pi_half2, pi_half2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        matrix_hat = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, matrix_hat, atol=atol, rtol=rtol)

    def test_corner_case(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(
            (
                (-0.7799533010, -0.5432914495, 0.3106555045),
                (0.0492402576, -0.5481169224, -0.8349509239),
                (0.6238971353, -0.6359263659, 0.4542570710),
            ),
            device=device,
            dtype=dtype,
        )
        quaternion_true = torch.tensor(
            (0.177614107728004, 0.280136495828629, -0.440902262926102, 0.834015488624573), device=device, dtype=dtype
        )
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        torch.set_printoptions(precision=10)
        self.assert_close(quaternion_true, quaternion, atol=atol, rtol=rtol)

    def test_cond1_180_rot_x(self, device, dtype, atol, rtol):
        # 180° rotation around X: trace < 0, m00 > m11 and m00 > m22 → activates cond_1 branch.
        # R_x(π) = diag(1, -1, -1); expected quaternion (w,x,y,z) = (0, 1, 0, 0).
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        self.assert_close(quaternion.abs(), expected.abs(), atol=atol, rtol=rtol)
        # Round-trip: convert back and verify the rotation matrix is recovered.
        mat_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(mat_back, matrix, atol=atol, rtol=rtol)

    def test_cond2_180_rot_y(self, device, dtype, atol, rtol):
        # 180° rotation around Y: trace < 0, m11 > m22 and m00 not dominant → activates cond_2 branch.
        # R_y(π) = diag(-1, 1, -1); expected quaternion (w,x,y,z) = (0, 0, 1, 0).
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        self.assert_close(quaternion.abs(), expected.abs(), atol=atol, rtol=rtol)
        mat_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(mat_back, matrix, atol=atol, rtol=rtol)

    def test_all_four_branches_in_batch(self, device, dtype, atol, rtol):
        # Batch of 4 rotation matrices that each activate a different internal branch.
        # Verify consistency via round-trip: R → q → R must recover the original rotation.
        eps = torch.finfo(dtype).eps
        identity = torch.eye(3, device=device, dtype=dtype)  # trace > 0 → trace_positive_cond
        rot_x_180 = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        rot_y_180 = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        rot_z_180 = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        batch = torch.stack([identity, rot_x_180, rot_y_180, rot_z_180])  # (4, 3, 3)
        quaternions = kornia.geometry.conversions.rotation_matrix_to_quaternion(batch, eps=eps)
        mats_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternions)
        self.assert_close(mats_back, batch, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        matrix = torch.eye(3, device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.rotation_matrix_to_quaternion, eps=eps), (matrix,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)

    def test_convention_w_is_not_canonicalised_to_non_negative(self, device, dtype):
        # Convention pin: rotation_matrix_to_quaternion picks ONE of the two quaternions that
        # represent the input rotation, and the rule is NOT "return w >= 0". The branch is selected
        # by the sign of the trace:
        #   trace > 0  -> the returned w is >= 0 (0/325 negative over random rotations);
        #   trace <= 0 -> the *dominant* of x, y, z is forced >= 0 and w may come back NEGATIVE
        #                 (96/181 such cases in the audit sweep).
        # So a caller that assumes a non-negative real part is wrong for every rotation past ~120
        # degrees. Both branches are pinned here, each with a rotation whose "natural" quaternion
        # has the opposite sign of w, which is exactly what a canonicalising implementation would
        # not reproduce.
        # Expected values are the true unit quaternions (computed with stdlib below), not the
        # function's own output: the returned components carry an extra ~1e-9 from the default
        # eps added inside the sqrt, which bare assert_close absorbs and no pin here asserts.
        # Snippet used to generate the matrices and the expected quaternions (stdlib only):
        #   import math
        #   n = math.sqrt(14.0)
        #   R = I + sin(theta) * K + (1 - cos(theta)) * K @ K   # Rodrigues, K = skew(axis)
        #   axis, theta = (1 / n, 2 / n, -3 / n), math.radians(170.0)   # trace -0.9696155060244165
        #     R -> [[-0.8430357706541933,  0.4227722475733091, -0.3324970918358584],
        #           [ 0.14431568185875046, -0.4177198235801487, -0.8970413217671823],
        #           [-0.5181348023122309, -0.8042224665289962,  0.29114008820992554]]
        #     the two representatives are +-[0.08715574274765814, 0.2662442321985726,
        #       0.5324884643971451, -0.7987326965957178]; |z| dominates, so the one with z >= 0
        #       is returned and its w is negative
        #   axis, theta = (1 / n, 2 / n, 3 / n), math.radians(60.0)     # trace 2.0
        #     R -> [[ 0.5357142857142858, -0.6229365034008422,  0.5700529070291328],
        #           [ 0.765793646257985,   0.6428571428571429, -0.01716931065742361],
        #           [-0.3557671927434186,  0.4457407392288521,  0.8214285714285714]]
        #     representatives +-[0.8660254037844387, 0.13363062095621217, 0.26726124191242434,
        #       0.40089186286863654]; the w >= 0 one is returned
        rot_trace_negative = torch.tensor(
            [
                [-0.8430357706541933, 0.4227722475733091, -0.3324970918358584],
                [0.14431568185875046, -0.4177198235801487, -0.8970413217671823],
                [-0.5181348023122309, -0.8042224665289962, 0.29114008820992554],
            ],
            device=device,
            dtype=dtype,
        )
        quaternion_trace_negative = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot_trace_negative)
        self.assert_close(
            quaternion_trace_negative,
            torch.tensor(
                [-0.08715574274765814, -0.2662442321985726, -0.5324884643971451, 0.7987326965957178],
                device=device,
                dtype=dtype,
            ),
        )
        assert quaternion_trace_negative[0] < 0.0
        assert quaternion_trace_negative[3] > 0.0

        rot_trace_positive = torch.tensor(
            [
                [0.5357142857142858, -0.6229365034008422, 0.5700529070291328],
                [0.765793646257985, 0.6428571428571429, -0.01716931065742361],
                [-0.3557671927434186, 0.4457407392288521, 0.8214285714285714],
            ],
            device=device,
            dtype=dtype,
        )
        quaternion_trace_positive = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot_trace_positive)
        self.assert_close(
            quaternion_trace_positive,
            torch.tensor(
                [0.8660254037844387, 0.13363062095621217, 0.26726124191242434, 0.40089186286863654],
                device=device,
                dtype=dtype,
            ),
        )
        assert quaternion_trace_positive[0] > 0.0


class TestQuaternionToRotationMatrix(BaseTester):
    @pytest.mark.parametrize("batch_dims", ((), (1,), (3,), (8,), (1, 1), (5, 6)))
    def test_smoke_batch(self, batch_dims, device, dtype):
        quaternion = torch.zeros(*batch_dims, 4, device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        assert matrix.shape == (*batch_dims, 3, 3)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=torch.float64)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_to_rotation_matrix), (quaternion,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_to_rotation_matrix
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)

    def test_convention_quaternion_component_order_is_w_x_y_z(self, device, dtype):
        # Convention pin -- THE trap of this module: quaternions are (w, x, y, z), real part FIRST.
        # (1, 0, 0, 0) is the identity. The (x, y, z, w) misreading of the same four numbers,
        # (0, 0, 0, 1), does not raise and does not return anything obviously wrong -- it returns
        # diag(-1, -1, 1), a perfectly valid 180-degree rotation about z. That is why the
        # counter-literal is pinned alongside the identity: an order swap is silent, and only the
        # second assertion catches it.
        # Snippet used to generate expected (stdlib only, R = I + 2*w*K + 2*K@K with K = skew(v)):
        #   q = (1, 0, 0, 0) -> v = 0, K = 0 -> R = I
        #   q = (0, 0, 0, 1) -> w = 0, v = (0, 0, 1)
        #     K @ K = diag(-1, -1, 0) -> R = I + 2 * diag(-1, -1, 0) = diag(-1, -1, 1)
        real_part_first = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        )
        # .to(dtype) because quaternion_to_rotation_matrix returns float32 for float16/bfloat16
        # inputs; the cast keeps this pin about the component order and nothing else.
        self.assert_close(real_part_first.to(dtype), torch.eye(3, device=device, dtype=dtype))

        read_as_xyzw = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
        )
        self.assert_close(
            read_as_xyzw.to(dtype),
            torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device, dtype=dtype)),
        )

    def test_convention_double_cover_q_and_minus_q_give_identical_matrices(self, device, dtype):
        # Convention pin: the unit quaternions double-cover SO(3), and every term of the rotation
        # matrix is a product of two quaternion components, so negating the whole quaternion leaves
        # the matrix BIT-identical -- not merely close. torch.equal, not assert_close: the measured
        # max difference is exactly 0.0, and the identity held in 500/500 random float64 draws
        # (seeded torch.Generator(6)) as well as at float32/float16/bfloat16 for the pinned input.
        # The input is non-unit and non-axis-aligned so the pin cannot pass by symmetry.
        quaternion = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)

        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)

        assert torch.equal(kornia.geometry.conversions.quaternion_to_rotation_matrix(-quaternion), rot)

    def test_convention_non_unit_quaternion_is_normalized_internally(self, device, dtype):
        # Convention pin: quaternion_to_rotation_matrix calls normalize_quaternion on its input
        # first, so a non-unit quaternion is accepted silently and yields the same rotation as its
        # normalised form -- the docstring never says so. Pinned two ways: the returned matrix
        # equals the one built from the unit quaternion (stdlib literal below), and rescaling the
        # input leaves the output bit-identical. The scale factors are powers of two so that the
        # scaling is exact in binary floating point at every dtype (0.001 is not: at bfloat16
        # 0.001 * q rounds differently and the matrices then differ by 1.6e-2).
        # Snippet used to generate expected (stdlib only, q = (1, 2, 3, 4) normalised):
        #   import math
        #   u = [v / math.sqrt(30.0) for v in (1.0, 2.0, 3.0, 4.0)]
        #   nv = math.sqrt(u[1] ** 2 + u[2] ** 2 + u[3] ** 2); theta = 2 * math.atan2(nv, u[0])
        #   Rodrigues([u[i + 1] / nv for i in range(3)], theta) ->
        #     [[-0.666666666666667,   0.13333333333333341, 0.7333333333333334],
        #      [ 0.6666666666666667, -0.3333333333333337,  0.6666666666666669],
        #      [ 0.3333333333333335,  0.9333333333333335,  0.1333333333333333]]
        quaternion = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)

        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)

        expected = torch.tensor(
            [
                [-0.666666666666667, 0.13333333333333341, 0.7333333333333334],
                [0.6666666666666667, -0.3333333333333337, 0.6666666666666669],
                [0.3333333333333335, 0.9333333333333335, 0.1333333333333333],
            ],
            device=device,
            dtype=dtype,
        )
        # .to(dtype) because the function returns float32 for float16/bfloat16 inputs.
        self.assert_close(rot.to(dtype), expected)

        assert torch.equal(kornia.geometry.conversions.quaternion_to_rotation_matrix(2.0 * quaternion), rot)
        assert torch.equal(kornia.geometry.conversions.quaternion_to_rotation_matrix(0.0009765625 * quaternion), rot)

    def test_convention_normalize_quaternion_is_l2_over_the_last_axis(self, device, dtype):
        # Convention pin (normalize_quaternion has no test class of its own; this is its nearest
        # sibling -- quaternion_to_rotation_matrix calls it on every input): it is a plain L2
        # normalisation of the last axis and nothing more. It does NOT reorder, and it does NOT
        # canonicalise the sign, so the whole vector keeps its sign. That makes it the one symbol
        # in this file whose "(x, y, z, w) or (w, x, y, z)" docstring phrasing is actually true:
        # the same four numbers in the other order come back scaled by the same factor, which the
        # third assertion pins.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   [v / math.sqrt(30.0) for v in (1.0, 2.0, 3.0, 4.0)]
        #     -> [0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214]
        expected = torch.tensor(
            [0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214],
            device=device,
            dtype=dtype,
        )

        out = kornia.geometry.conversions.normalize_quaternion(
            torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)
        )
        self.assert_close(out, expected)

        out_negated = kornia.geometry.conversions.normalize_quaternion(
            torch.tensor([-1.0, -2.0, -3.0, -4.0], device=device, dtype=dtype)
        )
        self.assert_close(out_negated, -expected)

        out_reversed = kornia.geometry.conversions.normalize_quaternion(
            torch.tensor([4.0, 3.0, 2.0, 1.0], device=device, dtype=dtype)
        )
        self.assert_close(out_reversed, expected.flip(0))


class TestQuaternionLogToExp(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion_log = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log)
        assert quaternion_exp.shape == (batch_size, 4)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), torch.sin(one), 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, torch.sin(one), 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, 0.0, torch.sin(one)), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)

        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        quaternion_log_hat = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, quaternion_log_hat, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_log_to_exp, eps=eps), (quaternion,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)

    def test_convention_exp_of_v_equals_axis_angle_to_quaternion_of_twice_v(self, device, dtype):
        # Convention pin: the log quaternion is (theta / 2) * axis, NOT the axis-angle vector, so
        # the exponential map is exactly axis_angle_to_quaternion applied to 2 * v. A caller that
        # feeds an axis-angle vector straight into quaternion_log_to_exp gets a rotation of half
        # the intended angle, silently. The size contract of the pair is pinned alongside:
        # (*, 3) -> (*, 4) here, (*, 4) -> (*, 3) in quaternion_exp_to_log.
        # Snippet used to generate expected (stdlib only, v = (0.15, 0.2, 0.25), theta = 2 * |v|):
        #   import math
        #   th = math.sqrt(0.15 ** 2 + 0.2 ** 2 + 0.25 ** 2) * 2   # 0.7071067811865476
        #   ax = [x / (th / 2) for x in (0.15, 0.2, 0.25)]
        #   [math.cos(th / 2)] + [math.sin(th / 2) * a for a in ax]
        #     -> [0.9381483350397287, 0.14689447322208307, 0.19585929762944412, 0.24482412203680515]
        # assert_close and not torch.equal: the two routes agree bit-for-bit at float64, float32
        # and float16 for this input, but that is an accident of the input -- over 500 random
        # float64 vectors (seeded torch.Generator(4)) only 142/500 are bit-identical (worst
        # difference 4.440892098500626e-16), and at bfloat16 the pinned input already differs by
        # 9.765625e-04 because ||2v|| / 2 and ||v|| round apart.
        log_quaternion = torch.tensor([0.15, 0.2, 0.25], device=device, dtype=dtype)

        out = kornia.geometry.conversions.quaternion_log_to_exp(log_quaternion)

        self.assert_close(
            out,
            torch.tensor(
                [0.9381483350397287, 0.14689447322208307, 0.19585929762944412, 0.24482412203680515],
                device=device,
                dtype=dtype,
            ),
        )
        self.assert_close(out, kornia.geometry.conversions.axis_angle_to_quaternion(2.0 * log_quaternion))

        assert kornia.geometry.conversions.quaternion_log_to_exp(
            torch.zeros(2, 5, 3, device=device, dtype=dtype)
        ).shape == (2, 5, 4)

    def test_convention_exp_real_part_is_cosine_of_the_norm(self, device, dtype):
        # Convention pin: the exponential map returns w = cos(||v||) and (x, y, z) = sin(||v||) * v
        # / ||v||, so it is NOT restricted to the w >= 0 half of the double cover: any ||v|| > pi/2
        # (i.e. any rotation past 180 degrees, since theta = 2 * ||v||) lands in the w < 0 half.
        # Pinned at ||v|| = 2 rad, where w is clearly negative; the output is still a unit
        # quaternion, which the norm assertion states.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.cos(2.0), math.sin(2.0) -> (-0.4161468365471424, 0.9092974268256817)
        out = kornia.geometry.conversions.quaternion_log_to_exp(
            torch.tensor([0.0, 0.0, 2.0], device=device, dtype=dtype)
        )

        self.assert_close(
            out,
            torch.tensor([-0.4161468365471424, 0.0, 0.0, 0.9092974268256817], device=device, dtype=dtype),
        )
        self.assert_close(out.norm(), torch.tensor(1.0, device=device, dtype=dtype))

    def test_convention_exp_of_log_is_the_identity_except_at_minus_one(self, device, dtype):
        # Convention pin (domain fact of the map, not a defect): quaternion_log_to_exp composed
        # with quaternion_exp_to_log is the identity, with exactly one exception -- the pure-real
        # quaternion (-1, 0, 0, 0), whose log is genuinely the origin in this parametrisation, so
        # the round-trip returns the OTHER half of the double cover, (1, 0, 0, 0). The sign of a
        # non-zero vector part is preserved, which is what the third case pins: (-1, 0, 0, -1)/
        # sqrt(2) comes back as itself and is not flipped to its positive-w twin.
        # Snippet used to generate expected (stdlib only):
        #   exp(log(q)) = q for every unit q except q = (-1, 0, 0, 0)
        #   1 / math.sqrt(2.0) -> 0.7071067811865476
        # float16 is skipped: quaternion_exp_to_log((-1, 0, 0, 0)) returns NaN there, so the
        # exception case cannot be evaluated at all (float64/float32/bfloat16 all return the
        # origin as documented above).
        if dtype == torch.float16:
            pytest.skip("quaternion_exp_to_log((-1, 0, 0, 0)) is NaN at float16, so exp(log(.)) is undefined")

        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        exp_to_log = kornia.geometry.conversions.quaternion_exp_to_log
        log_to_exp = kornia.geometry.conversions.quaternion_log_to_exp

        self.assert_close(log_to_exp(exp_to_log(identity)), identity)
        self.assert_close(log_to_exp(exp_to_log(-identity)), identity)

        half_turn = torch.tensor([-0.7071067811865476, 0.0, 0.0, -0.7071067811865476], device=device, dtype=dtype)
        self.assert_close(log_to_exp(exp_to_log(half_turn)), half_turn)


class TestQuaternionExpToLog(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        assert quaternion_log.shape == (batch_size, 3)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        quaternion_exp_hat = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, quaternion_exp_hat, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_exp_to_log, eps=eps), (quaternion,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_exp_to_log
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)

    def test_convention_log_is_half_the_axis_angle_on_the_w_positive_half(self, device, dtype):
        # Convention pin: the log quaternion is (theta / 2) * axis, i.e. exactly half the
        # axis-angle vector -- so quaternion_exp_to_log(q) == quaternion_to_axis_angle(q) / 2, but
        # ONLY on the w >= 0 half. The two functions treat the double cover differently:
        # quaternion_to_axis_angle collapses q and -q onto the same |theta| <= pi vector, while
        # quaternion_exp_to_log takes acos(w) at face value and returns (pi - theta/2) along the
        # negated axis for w < 0. The second half of this pin states that divergence explicitly,
        # because "log is half the axis-angle" is false without the restriction: over 500 random
        # float64 unit quaternions (seeded torch.Generator(3)) the two agree to 4.44e-16 on the
        # 243 with w >= 0 and disagree by up to 3.1367975802888637 on the 257 with w < 0.
        # The size contract (*, 4) -> (*, 3) is pinned alongside.
        # Snippet used to generate expected (stdlib only, axis_angle = (0.3, 0.4, 0.5)):
        #   import math
        #   v = [0.15, 0.2, 0.25]                       # = axis_angle / 2, the expected log
        #   nv = math.sqrt(sum(x * x for x in v))       # 0.3535533905932738
        #   q = [math.cos(nv)] + [math.sin(nv) * x / nv for x in v]
        #     -> [0.9381483350397287, 0.14689447322208307, 0.19585929762944412, 0.24482412203680515]
        #   for -q the log is (nv - pi) * axis:
        #   [-(math.pi - nv) * (x / nv) for x in v]
        #     -> [-1.1828648814475096, -1.5771531752633463, -1.9714414690791828]
        quaternion = torch.tensor(
            [0.9381483350397287, 0.14689447322208307, 0.19585929762944412, 0.24482412203680515],
            device=device,
            dtype=dtype,
        )

        out = kornia.geometry.conversions.quaternion_exp_to_log(quaternion)

        self.assert_close(out, torch.tensor([0.15, 0.2, 0.25], device=device, dtype=dtype))
        self.assert_close(out, kornia.geometry.conversions.quaternion_to_axis_angle(quaternion) / 2.0)

        out_negated = kornia.geometry.conversions.quaternion_exp_to_log(-quaternion)
        self.assert_close(
            out_negated,
            torch.tensor([-1.1828648814475096, -1.5771531752633463, -1.9714414690791828], device=device, dtype=dtype),
        )

        assert kornia.geometry.conversions.quaternion_exp_to_log(
            torch.zeros(2, 5, 4, device=device, dtype=dtype)
        ).shape == (2, 5, 3)

    def test_convention_log_of_exp_is_exact_below_pi_and_wraps_above(self, device, dtype):
        # Convention pin (domain fact of the map, not a defect): quaternion_exp_to_log composed
        # with quaternion_log_to_exp reproduces its input only for 0 < ||v|| < pi. Above pi the
        # rotation has passed a full turn (theta = 2 * ||v||) and the result wraps into
        # ||v|| - 2*pi, i.e. it comes back with the OPPOSITE sign, which the second case pins:
        # a caller doing exp/log arithmetic on large vectors must reduce the norm itself.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   the round-trip is the identity for ||v|| < pi -> [0.0, 0.0, 1.0]
        #   math.pi + 0.5 - 2 * math.pi -> -2.641592653589793
        exp_to_log = kornia.geometry.conversions.quaternion_exp_to_log
        log_to_exp = kornia.geometry.conversions.quaternion_log_to_exp

        below_pi = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        self.assert_close(exp_to_log(log_to_exp(below_pi)), below_pi)

        above_pi = torch.tensor([0.0, 0.0, 3.641592653589793], device=device, dtype=dtype)
        self.assert_close(
            exp_to_log(log_to_exp(above_pi)),
            torch.tensor([0.0, 0.0, -2.641592653589793], device=device, dtype=dtype),
        )

    def test_convention_log_of_exp_collapses_to_zero_at_pi_in_float64(self, device):
        # Convention pin (domain fact of the map): at exactly ||v|| = pi the exponential map lands
        # on (-1, 0, 0, 0), whose log genuinely IS the origin in this parametrisation, so the
        # round-trip collapses to ~0 instead of returning pi -- the one interior point where the
        # log/exp pair is not invertible. float64 is hardcoded and the dtype fixture dropped
        # because the collapse needs cos(||v||) to round to exactly -1 and the vector part to fall
        # below the eps clamp, which only happens at float64: at float32 the same input returns
        # -3.1415927410125732 (no collapse), at float16/bfloat16 3.140625. MPS is skipped visibly
        # because it has no float64 at all.
        # Snippet used to generate expected (stdlib only):
        #   math.cos(math.pi), math.sin(math.pi) -> (-1.0, 1.2246467991473532e-16)
        # Measured round-trip value at float64 (torch 2.9.1, cpu): 3.847341387443579e-08, i.e. an
        # error of 3.14 against the input, so atol 1e-7 pins the collapse without pinning the
        # residue itself.
        if device.type == "mps":
            pytest.skip("MPS has no float64, and this collapse is float64-only by construction")

        at_pi = torch.tensor([0.0, 0.0, 3.141592653589793], device=device, dtype=torch.float64)

        out = kornia.geometry.conversions.quaternion_exp_to_log(
            kornia.geometry.conversions.quaternion_log_to_exp(at_pi)
        )

        self.assert_close(out, torch.zeros(3, device=device, dtype=torch.float64), atol=1e-7, rtol=0.0)


class TestAngleAxisToRotationMatrix(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_axis_angle_gradcheck(self, batch_size, device, atol, rtol):
        dtype = torch.float64
        # generate input data
        axis_angle = torch.rand(batch_size, 3, device=device, dtype=dtype)
        eye_batch = eye_like(3, axis_angle)

        # apply transform
        rotation_matrix = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)

        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        self.assert_close(rotation_matrix_eye, eye_batch, atol=atol, rtol=rtol)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.axis_angle_to_rotation_matrix, (axis_angle,))

    def test_axis_angle_to_rotation_matrix(self, device, dtype, atol, rtol):
        rmat_1 = torch.tensor(
            (
                (-0.30382753, -0.95095137, -0.05814062),
                (-0.71581715, 0.26812278, -0.64476041),
                (0.62872461, -0.15427791, -0.76217038),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_1 = torch.tensor((1.50485376, -2.10737739, 0.7214174), device=device, dtype=dtype)

        rmat_2 = torch.tensor(
            (
                (0.6027768, -0.79275544, -0.09054801),
                (-0.67915707, -0.56931658, 0.46327563),
                (-0.41881476, -0.21775548, -0.88157628),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_2 = torch.tensor((-2.44916812, 1.18053411, 0.4085298), device=device, dtype=dtype)
        rmat = torch.stack((rmat_2, rmat_1), dim=0)
        rvec = torch.stack((rvec_2, rvec_1), dim=0)

        self.assert_close(kornia.geometry.conversions.axis_angle_to_rotation_matrix(rvec), rmat, atol=atol, rtol=rtol)

    def test_convention_positive_angle_about_z_maps_x_to_y(self, device, dtype):
        # Convention pin (covers quaternion_to_rotation_matrix too -- both routes to a rotation
        # matrix in this module must agree): rotations follow the right-hand rule, so a positive
        # angle about +z takes x_hat to y_hat and the matrix is
        # [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], NOT its transpose. Pinned at theta = 0.6 rad
        # rather than a quarter turn so that a transposed or sign-flipped implementation cannot
        # slip through on symmetry, and the mapped basis vector is asserted as well as the matrix
        # so the claim is stated the way a reader will use it.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.cos(0.6), math.sin(0.6) -> (0.8253356149096783, 0.5646424733950354)
        #   the same rotation as a quaternion, (cos(0.3), 0, 0, sin(0.3))
        #     -> (0.955336489125606, 0.0, 0.0, 0.29552020666133955)
        expected = torch.tensor(
            [
                [0.8253356149096783, -0.5646424733950354, 0.0],
                [0.5646424733950354, 0.8253356149096783, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        expected_x_maps_to = torch.tensor([0.8253356149096783, 0.5646424733950354, 0.0], device=device, dtype=dtype)
        x_hat = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)

        rot_from_axis_angle = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            torch.tensor([[0.0, 0.0, 0.6]], device=device, dtype=dtype)
        )[0]
        self.assert_close(rot_from_axis_angle, expected)
        self.assert_close(rot_from_axis_angle @ x_hat, expected_x_maps_to)

        rot_from_quaternion = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            torch.tensor([0.955336489125606, 0.0, 0.0, 0.29552020666133955], device=device, dtype=dtype)
        )
        # .to(dtype) because quaternion_to_rotation_matrix returns float32 for float16/bfloat16
        # inputs; the cast keeps this pin about the rotation sense and nothing else.
        self.assert_close(rot_from_quaternion.to(dtype), expected)
        self.assert_close(rot_from_quaternion.to(dtype) @ x_hat, expected_x_maps_to)

    def test_convention_axis_angle_is_in_radians(self, device, dtype):
        # Convention pin: the axis-angle vector's magnitude is an angle in RADIANS. This is the
        # trap that separates this family from angle_to_rotation_matrix in the same module, which
        # reads DEGREES (see TestRadDegConversions.test_convention_angle_to_rotation_matrix_takes_
        # degrees) -- the two live a few hundred lines apart and neither says so in its signature.
        # pi/2 gives the quarter turn; feeding 90 in the belief that it is degrees gives cos/sin of
        # 90 radians instead, a rotation of roughly 152 degrees that is nowhere near a quarter turn.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.cos(math.pi / 2), math.sin(math.pi / 2) -> (6.123233995736766e-17, 1.0)
        #   math.cos(90.0), math.sin(90.0) -> (-0.4480736161291702, 0.8939966636005579)
        quarter_turn = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            torch.tensor([[0.0, 0.0, torch.pi / 2]], device=device, dtype=dtype)
        )[0]
        self.assert_close(
            quarter_turn,
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                device=device,
                dtype=dtype,
            ),
        )

        read_as_degrees = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            torch.tensor([[0.0, 0.0, 90.0]], device=device, dtype=dtype)
        )[0]
        self.assert_close(
            read_as_degrees,
            torch.tensor(
                [
                    [-0.4480736161291702, -0.8939966636005579, 0.0],
                    [0.8939966636005579, -0.4480736161291702, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            ),
        )


class TestRotationMatrixToAngleAxis(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_quaternion_gradcheck(self, batch_size, device, dtype, atol, rtol):
        # generate input data
        quaternion = torch.rand(batch_size, 4, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.normalize_quaternion(quaternion + 1e-6)
        rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion=quaternion)

        eye_batch = eye_like(3, rotation_matrix)
        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        # This didn't pass with atol=0.001, rtol=0.001 for float16 Cuda 11.2 GeForce 1080 Ti
        self.assert_close(rotation_matrix_eye, eye_batch, atol=atol * 10.0, rtol=rtol * 10.0)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device):
        dtype = torch.float64
        quaternion = torch.rand(batch_size, 4, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.normalize_quaternion(quaternion + 1e-6)
        rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion=quaternion)
        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.rotation_matrix_to_axis_angle, (rotation_matrix,))

    def test_rotation_matrix_to_axis_angle(self, device, dtype, atol, rtol):
        rmat_1 = torch.tensor(
            (
                (-0.30382753, -0.95095137, -0.05814062),
                (-0.71581715, 0.26812278, -0.64476041),
                (0.62872461, -0.15427791, -0.76217038),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_1 = torch.tensor((1.50485376, -2.10737739, 0.7214174), device=device, dtype=dtype)

        rmat_2 = torch.tensor(
            (
                (0.6027768, -0.79275544, -0.09054801),
                (-0.67915707, -0.56931658, 0.46327563),
                (-0.41881476, -0.21775548, -0.88157628),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_2 = torch.tensor((-2.44916812, 1.18053411, 0.4085298), device=device, dtype=dtype)
        rmat = torch.stack((rmat_2, rmat_1), dim=0)
        rvec = torch.stack((rvec_2, rvec_1), dim=0)

        self.assert_close(kornia.geometry.conversions.rotation_matrix_to_axis_angle(rmat), rvec, atol=atol, rtol=rtol)

    def test_convention_accepts_any_leading_batch_dimensions(self, device, dtype):
        # Convention pin (rotation_matrix_to_axis_angle has no test class under its own name; this
        # class is the one that exercises it): the shape contract is the full (*, 3, 3) -> (*, 3),
        # not the (N, 3, 3) -> (N, 3) its docstring states. An unbatched (3, 3) works -- that is
        # what its own doctest passes -- and so does any number of leading batch dimensions.
        # Expected is the true axis-angle vector computed with stdlib, not the function's output.
        # Snippet used to generate the matrix and expected (stdlib only):
        #   import math
        #   n = math.sqrt(14.0); axis = (1 / n, 2 / n, -3 / n); theta = math.radians(170.0)
        #   R = I + sin(theta) * K + (1 - cos(theta)) * K @ K    # Rodrigues, K = skew(axis)
        #   [theta * a for a in axis]
        #     -> [0.7929800678379483, 1.5859601356758966, -2.378940203513845]
        rot = torch.tensor(
            [
                [-0.8430357706541933, 0.4227722475733091, -0.3324970918358584],
                [0.14431568185875046, -0.4177198235801487, -0.8970413217671823],
                [-0.5181348023122309, -0.8042224665289962, 0.29114008820992554],
            ],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor(
            [0.7929800678379483, 1.5859601356758966, -2.378940203513845], device=device, dtype=dtype
        )

        unbatched = kornia.geometry.conversions.rotation_matrix_to_axis_angle(rot)
        assert unbatched.shape == (3,)
        self.assert_close(unbatched, expected)

        multi_batched = kornia.geometry.conversions.rotation_matrix_to_axis_angle(rot.expand(2, 5, 3, 3))
        assert multi_batched.shape == (2, 5, 3)
        self.assert_close(multi_batched[1, 4], expected)

    def test_convention_axis_angle_roundtrip_tolerance_is_1e_6_in_float64(self, device):
        # Convention pin: rotation_matrix_to_axis_angle composed with
        # axis_angle_to_rotation_matrix recovers the vector only to ~1e-6, and that floor does not
        # move with the dtype -- it is still ~1e-6 in float64, six orders worse than the machine
        # epsilon a reader would expect from a "round-trip" and eleven orders worse than the
        # quaternion leg (see TestAngleAxisToQuaternion.test_convention_axis_angle_quaternion_
        # roundtrip_is_exact_in_float64, which is exact at the same angles). Anyone comparing
        # rotations through this pair must budget 1e-6. float64 is hardcoded and the dtype fixture
        # dropped because the claim is precisely that float64 does NOT help; MPS is skipped visibly
        # because it has no float64 at all. The tolerance is the observed one and must not be
        # tightened.
        # Snippet used to generate the inputs (stdlib only):
        #   import math
        #   n = math.sqrt(14.0); axis = (1 / n, 2 / n, 3 / n)
        #   [[theta * a for a in axis] for theta in (1e-3, 0.7, 2.0, math.pi)]
        # Measured max |roundtrip - input| at those four thetas (torch 2.9.1, cpu float64):
        #   8.009844122083441e-07, 6.410323137862051e-07, 5.134006499929455e-07,
        #   5.387205765927661e-07 -- so atol 1e-6 clears the worst of them by 20%.
        if device.type == "mps":
            pytest.skip("MPS has no float64, and this pin is float64-only by construction")

        axis_angle = torch.tensor(
            [
                [0.0002672612419124244, 0.0005345224838248488, 0.0008017837257372733],
                [0.18708286933869706, 0.3741657386773941, 0.5612486080160912],
                [0.5345224838248488, 1.0690449676496976, 1.6035674514745464],
                [0.839625954181357, 1.679251908362714, 2.518877862544071],
            ],
            device=device,
            dtype=torch.float64,
        )

        roundtrip = kornia.geometry.conversions.rotation_matrix_to_axis_angle(
            kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)
        )

        self.assert_close(roundtrip, axis_angle, atol=1e-6, rtol=0.0)


class TestRadDegConversions(BaseTester):
    def test_pi(self):
        self.assert_close(kornia.constants.pi.item(), 3.141592)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_rad2deg(self, batch_shape, device, dtype):
        # generate input data
        x_rad = kornia.constants.pi * torch.rand(batch_shape, device=device, dtype=dtype)

        # convert radians/degrees
        x_deg = kornia.geometry.conversions.rad2deg(x_rad)
        x_deg_to_rad = kornia.geometry.conversions.deg2rad(x_deg)

        # compute error
        self.assert_close(x_rad, x_deg_to_rad)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_rad2deg_gradcheck(self, batch_shape, device):
        dtype = torch.float64
        x_rad = torch.rand(batch_shape, device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.rad2deg, (x_rad,))

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_deg2rad(self, batch_shape, device, dtype, atol, rtol):
        # generate input data
        x_deg = 180.0 * torch.rand(batch_shape, device=device, dtype=dtype)

        # convert radians/degrees
        x_rad = kornia.geometry.conversions.deg2rad(x_deg)
        x_rad_to_deg = kornia.geometry.conversions.rad2deg(x_rad)

        self.assert_close(x_deg, x_rad_to_deg, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_deg2rad_gradcheck(self, batch_shape, device):
        x_deg = 180.0 * torch.rand(batch_shape, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.conversions.deg2rad, (x_deg,))

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="kornia.constants.pi is float32, so f64 loses ~7 digits (angle_to_rotation_matrix "
        "inherits it via deg2rad) — kornia#3937",
        strict=True,
    )
    @pytest.mark.parametrize(
        ("op_name", "arg", "expected"),
        [
            ("rad2deg", torch.pi, 180.0),
            ("deg2rad", 180.0, torch.pi),
            ("angle_to_rotation_matrix", 90.0, [[0.0, 1.0], [-1.0, 0.0]]),
        ],
    )
    def test_convention_float64_results_are_exact_3937(self, device, op_name, arg, expected):
        # Intended behavior: each op is exact to the precision of its input dtype, like
        # torch.rad2deg / torch.deg2rad; angle_to_rotation_matrix(90) is then the exact quarter
        # turn. It is not: all three multiply by kornia.constants.pi, a *float32* tensor merely
        # cast to the input dtype, so a float64 input carries a systematic ~2.8e-8 relative
        # error (#3937). float64 is hardcoded (like test_rad2deg_gradcheck above) because at
        # float32 the biased constant *is* the correctly rounded pi; MPS is skipped visibly
        # below because it has no float64 at all, so without the skip the xfail would be
        # satisfied by a TypeError instead of the precision assert it documents (hence also
        # raises=AssertionError on the mark). Marked xfail(strict=True) so fixing #3937 makes
        # every case XPASS and forces this mark out — a one-place edit.
        # Snippet used to generate expected (stdlib + torch):
        #   math.degrees(math.pi) == 180.0 and (180.0 * math.pi) / 180.0 == math.pi exactly
        #   kornia rad2deg(tensor(pi, f64)).item()   -> 179.99999499104382
        #   kornia deg2rad(tensor(180., f64)).item() -> 3.1415927410125732 (math.pi + 8.7e-08)
        #   kornia angle_to_rotation_matrix(tensor(90., f64)).flatten().tolist() ->
        #     [-4.371139000186241e-08, 0.999999999999999, -0.999999999999999, -4.371139e-08]
        # atol/rtol 1e-12 sits between the current ~4.4e-8 cosine error and the 6.123234e-17
        # an unbiased constant would give.
        if device.type == "mps":
            pytest.skip("MPS has no float64, and this pin is float64-only by construction")

        op = getattr(kornia.geometry.conversions, op_name)

        out = op(torch.tensor(arg, device=device, dtype=torch.float64))

        self.assert_close(out, torch.tensor(expected, device=device, dtype=torch.float64), atol=1e-12, rtol=1e-12)

    def test_convention_angle_to_rotation_matrix_takes_degrees(self, device, dtype):
        # Convention pin: angle_to_rotation_matrix reads its argument in DEGREES (not radians)
        # and returns [[cos, sin], [-sin, cos]] -- the transpose of the textbook math-frame CCW
        # matrix. Pinned on a small non-symmetric angle so a sign flip on the off-diagonal is
        # caught.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
        #   [[c, s], [-s, c]] -> [[0.8660254037844387, 0.49999999999999994],
        #                         [-0.49999999999999994, 0.8660254037844387]]
        out = kornia.geometry.conversions.angle_to_rotation_matrix(torch.tensor(30.0, device=device, dtype=dtype))
        expected = torch.tensor([[0.8660254, 0.5], [-0.5, 0.8660254]], device=device, dtype=dtype)
        self.assert_close(out, expected)

        # A radian-reading implementation would turn pi/2 into the quarter turn [[0, 1], [-1, 0]];
        # this one reads pi/2 as 1.5708 *degrees* and returns a near-identity matrix instead.
        # Snippet used to generate expected:
        #   c, s = math.cos(math.radians(math.pi / 2)), math.sin(math.radians(math.pi / 2))
        #   [[c, s], [-s, c]] -> [[0.9996242168385687, 0.027412134354665284], ...]
        out_rad = kornia.geometry.conversions.angle_to_rotation_matrix(
            torch.tensor(torch.pi / 2, device=device, dtype=dtype)
        )
        expected_rad = torch.tensor([[0.99962422, 0.02741213], [-0.02741213, 0.99962422]], device=device, dtype=dtype)
        self.assert_close(out_rad, expected_rad)

    @pytest.mark.parametrize(
        ("op_name", "arg", "expected"),
        [
            ("rad2deg", [1, 2, 3], [60.0, 120.0, 180.0]),
            ("deg2rad", [180, 90], [3.0, 1.5]),
            ("angle_to_rotation_matrix", [90], [[[0.07073720, 0.99749500], [-0.99749500, 0.07073720]]]),
        ],
    )
    def test_wart_integer_input_truncates_pi_to_3_3937(self, device, op_name, arg, expected):
        # Wart pins for #3937: assert the CURRENT broken outputs the docstring warnings document.
        # kornia.constants.pi is cast to the *integer* input dtype and truncates to 3, so rad2deg
        # divides by 3, deg2rad multiplies by 3 (90 degrees -> 1.5 radians), and the downstream
        # angle_to_rotation_matrix([90]) is nowhere near the quarter turn. If a case fails, #3937
        # was (partly) fixed -- update or remove the warnings in rad2deg, deg2rad and
        # angle_to_rotation_matrix and flip/remove the strict xfail above. NOT a contract that
        # int inputs must keep these values: what they *should* do (promote to float like
        # torch.rad2deg, or raise) is a maintainer decision, and a strict xfail asserting the
        # promoted-float answer would stay silently XFAIL forever if the fix chose to raise;
        # a wart pin flips loudly under either polarity.
        # Snippet used to generate expected (torch only):
        #   kornia rad2deg(torch.tensor([1, 2, 3])) -> tensor([ 60., 120., 180.]), dtype float32
        #     (torch.rad2deg gives [ 57.2958, 114.5916, 171.8873])
        #   kornia deg2rad(torch.tensor([180, 90])) -> tensor([3.0000, 1.5000]), dtype float32
        #     (torch.deg2rad gives [3.1416, 1.5708])
        #   kornia angle_to_rotation_matrix(torch.tensor([90])).flatten().tolist() ->
        #     [0.07073719799518585, 0.9974949955940247, -0.9974949955940247, 0.07073719799518585]
        #     (math.cos(1.5), math.sin(1.5) -> (0.0707372016677029, 0.9974949866040544))
        op = getattr(kornia.geometry.conversions, op_name)

        out = op(torch.tensor(arg, device=device))

        assert out.dtype == torch.float32
        self.assert_close(out, torch.tensor(expected, device=device, dtype=torch.float32), atol=1e-4, rtol=1e-4)


class TestPolCartConversions(BaseTester):
    def test_smoke(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        assert kornia.geometry.conversions.pol2cart(x, x) is not None
        assert kornia.geometry.conversions.cart2pol(x, x) is not None

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_pol2cart(self, batch_shape, device, dtype):
        # generate input data
        rho = torch.rand(batch_shape, dtype=dtype)
        phi = kornia.constants.pi * torch.rand(batch_shape, dtype=dtype)
        rho = rho.to(device)
        phi = phi.to(device)

        # convert pol/cart
        x_pol2cart, y_pol2cart = kornia.geometry.conversions.pol2cart(rho, phi)
        rho_pol2cart, phi_pol2cart = kornia.geometry.conversions.cart2pol(x_pol2cart, y_pol2cart, 0)

        self.assert_close(rho, rho_pol2cart)
        self.assert_close(phi, phi_pol2cart)

    @pytest.mark.parametrize("batch_shape", [(2, 3)])
    def test_gradcheck(self, batch_shape, device):
        rho = torch.rand(batch_shape, dtype=torch.float64, device=device)
        phi = kornia.constants.pi * torch.rand(batch_shape, dtype=torch.float64, device=device)
        self.gradcheck(kornia.geometry.conversions.pol2cart, (rho, phi))
        self.gradcheck(kornia.geometry.conversions.cart2pol, (rho, phi))

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_cart2pol(self, batch_shape, device, dtype):
        # generate input data
        x = torch.rand(batch_shape, dtype=dtype)
        y = torch.rand(batch_shape, dtype=dtype)
        x = x.to(device)
        y = y.to(device)

        # convert cart/pol
        rho_cart2pol, phi_cart2pol = kornia.geometry.conversions.cart2pol(x, y, 0)
        x_cart2pol, y_cart2pol = kornia.geometry.conversions.pol2cart(rho_cart2pol, phi_cart2pol)

        self.assert_close(x, x_cart2pol)
        self.assert_close(y, y_cart2pol)

    def test_convention_pol2cart_takes_rho_phi_returns_x_y(self, device, dtype):
        # Convention pin: pol2cart's argument order is (rho, phi) and its return order is
        # (x, y), with phi in RADIANS measured from the +x axis: x = rho*cos(phi),
        # y = rho*sin(phi). The literal is deliberately off-axis (3 != 4) so that swapping
        # either the arguments or the returns is caught.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   phi = math.atan2(4.0, 3.0)  # 0.9272952180016122 rad
        #   5.0 * math.cos(phi), 5.0 * math.sin(phi) -> (3.0000000000000004, 4.0)
        rho = torch.tensor(5.0, device=device, dtype=dtype)
        phi = torch.tensor(0.9272952180016122, device=device, dtype=dtype)

        x, y = kornia.geometry.conversions.pol2cart(rho, phi)

        self.assert_close(x, torch.tensor(3.0, device=device, dtype=dtype))
        self.assert_close(y, torch.tensor(4.0, device=device, dtype=dtype))

    def test_convention_cart2pol_takes_x_y_and_phi_is_atan2_y_x(self, device, dtype):
        # Convention pin: cart2pol's argument order is (x, y) and its return order is
        # (rho, phi), with phi = atan2(y, x) in radians -- zero on the +x axis and increasing
        # toward +y (which is clockwise *as displayed* under kornia's y-down image axes).
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.hypot(3.0, 4.0), math.atan2(4.0, 3.0) -> (5.0, 0.9272952180016122)
        #   math.atan2(1.0, 0.0) -> 1.5707963267948966  (atan2(x, y) would give 0.0 here)
        rho, phi = kornia.geometry.conversions.cart2pol(
            torch.tensor(3.0, device=device, dtype=dtype), torch.tensor(4.0, device=device, dtype=dtype)
        )
        self.assert_close(rho, torch.tensor(5.0, device=device, dtype=dtype))
        self.assert_close(phi, torch.tensor(0.9272952180016122, device=device, dtype=dtype))

        phi_y_axis = kornia.geometry.conversions.cart2pol(
            torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
        )[1]
        self.assert_close(phi_y_axis, torch.tensor(1.5707963267948966, device=device, dtype=dtype))

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="cart2pol returns sqrt(x**2 + y**2 + eps), biasing rho — kornia#3939",
        strict=True,
    )
    def test_convention_cart2pol_rho_is_the_exact_radius(self, device, dtype):
        # Intended behavior: rho is the Euclidean radius, so rho(0, 0) == 0. It currently is
        # not: eps is added *inside* the sqrt, so rho = sqrt(x**2 + y**2 + eps) and the origin
        # maps to sqrt(1e-8) = 1e-4 (see #3939; eps belongs in the gradient path, not the
        # value). Marked xfail(strict=True) so fixing #3939 makes this XPASS loudly.
        # Snippet used to generate expected (stdlib only):
        #   math.hypot(0.0, 0.0) -> 0.0 ; kornia cart2pol(0., 0.)[0].item() -> 0.0001
        if dtype == torch.float16:
            pytest.skip("float16 cannot represent the default eps=1e-8, so the bias is invisible there")

        rho = kornia.geometry.conversions.cart2pol(
            torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
        )[0]
        self.assert_close(rho, torch.tensor(0.0, device=device, dtype=dtype), atol=1e-6, rtol=0.0)

    def test_wart_rho_is_biased_by_eps_inside_the_sqrt_3939(self, device, dtype):
        # Wart pin for kornia#3939, companion to the strict xfail above: assert the CURRENT
        # biased rho. The xfail pins the intended rho(0, 0) == 0 but cannot flip under every fix
        # polarity -- the equally standard sqrt(clamp(x**2 + y**2, min=eps)) (the shape
        # normalize_pixel_coordinates already uses) also returns 1e-4 at the origin, leaving the
        # mark silently XFAIL with a stale reason string. So two cells are pinned: the origin,
        # rho = sqrt(eps) = 1e-4, which flips under a grad-only eps (rho 0) and under eps**2
        # inside the sqrt (rho 1e-8); and a sub-eps point x = 5e-5, whose x**2 = 2.5e-9 < eps
        # gives rho = sqrt(1.25e-8) ~ 1.118e-4, which additionally flips under the clamp shape
        # (rho 1e-4, 10.6 % below, outside rtol 1e-2). If either assert fails, #3939 was
        # (partly) fixed -- update or remove the warning in cart2pol and flip/remove the strict
        # xfail above. eps=1e-8 is passed explicitly so the pinned literals do not silently
        # track a later change to the default.
        # Snippet used to generate expected (torch only, executed at each pinned dtype):
        #   c2p = kornia.geometry.conversions.cart2pol
        #   c2p(torch.tensor(0., dtype=torch.float64), torch.tensor(0., dtype=torch.float64),
        #       eps=1e-8)[0] -> 0.0001                    (f32: 9.999999747378752e-05)
        #   c2p(torch.tensor(5e-5, dtype=torch.float64), torch.tensor(0., dtype=torch.float64),
        #       eps=1e-8)[0] -> 0.00011180339887498949    (f32: 0.00011180339788552374)
        # At bfloat16 the outputs land within 0.3 % of the literals (1.00136e-4, 1.12057e-4),
        # inside rtol 1e-2, so the pin holds there too.
        if dtype == torch.float16:
            pytest.skip("float16 cannot represent eps=1e-8, so rho is 0 at both pinned points and the bias invisible")

        zero = torch.tensor(0.0, device=device, dtype=dtype)

        rho_origin = kornia.geometry.conversions.cart2pol(zero, zero, eps=1e-8)[0]
        rho_sub_eps = kornia.geometry.conversions.cart2pol(
            torch.tensor(5e-5, device=device, dtype=dtype), zero, eps=1e-8
        )[0]

        self.assert_close(rho_origin, torch.tensor(1e-4, device=device, dtype=dtype), atol=0.0, rtol=1e-2)
        self.assert_close(
            rho_sub_eps, torch.tensor(1.1180339887498949e-4, device=device, dtype=dtype), atol=0.0, rtol=1e-2
        )

    def test_convention_positive_rotation_decreases_cart2pol_phi(self, device, dtype):
        # Cross-symbol convention pin: enforces the opposite-sense relation between
        # angle_to_rotation_matrix and cart2pol stated canonically in cart2pol's Convention
        # block (phi decreases by theta modulo 2*pi).
        # First case: no branch-cut crossing, so the raw difference is -theta itself.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   -math.radians(30.0) -> -0.5235987755982988
        v = torch.tensor([3.0, 4.0], device=device, dtype=dtype)
        phi0 = kornia.geometry.conversions.cart2pol(v[0], v[1])[1]

        rot = kornia.geometry.conversions.angle_to_rotation_matrix(torch.tensor(30.0, device=device, dtype=dtype))
        v_rot = rot @ v
        phi1 = kornia.geometry.conversions.cart2pol(v_rot[0], v_rot[1])[1]

        expected_delta = torch.tensor(-0.5235987755982988, device=device, dtype=dtype)
        self.assert_close(phi1 - phi0, expected_delta)

        # Second case: crossing the -x branch cut, where the returned phi is re-wrapped into
        # [-pi, pi] and only the difference modulo 2*pi is -theta (the worked -170 + 30 example
        # lives in cart2pol's Convention block).
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   5 * math.cos(math.radians(-170.0)), 5 * math.sin(math.radians(-170.0))
        #     -> (-4.92403876506104, -0.8682408883346514)
        #   math.radians(160.0) -> 2.792526803190927
        w = torch.tensor([-4.9240388, -0.8682409], device=device, dtype=dtype)
        phi0_cut = kornia.geometry.conversions.cart2pol(w[0], w[1])[1]

        w_rot = rot @ w
        phi1_cut = kornia.geometry.conversions.cart2pol(w_rot[0], w_rot[1])[1]

        self.assert_close(phi1_cut, torch.tensor(2.7925268, device=device, dtype=dtype))

        raw_delta = phi1_cut - phi0_cut
        wrapped_delta = torch.atan2(torch.sin(raw_delta), torch.cos(raw_delta))
        # The re-wrap atan2(sin, cos) adds two more transcendental roundings on top of the two
        # atan2 outputs it differences, overshooting the central per-dtype tolerances in the half
        # dtypes. Measured against the dtype-cast expected tensor the assert compares with
        # (-0.5234375 in both halves): |err| is 1.953125e-3 in float16 (wrapped -0.525390625;
        # central allowance atol 1e-3 + rtol 1e-3 * 0.52 = 1.52e-3) and 1.171875e-2 in bfloat16
        # (wrapped -0.53515625; allowance 1.19e-2 -- a 1.4 % margin that torch rounding drift
        # could erase). atol 2.4e-2 is ~2x the bfloat16 error; a sign-flipped or unwrapped delta
        # would still be off by >= 1.0.
        wrap_tol = {"atol": 2.4e-2, "rtol": 0.0} if dtype in (torch.float16, torch.bfloat16) else {}
        self.assert_close(wrapped_delta, expected_delta, **wrap_tol)


class TestConvertPointsToHomogeneous(BaseTester):
    def test_convert_points(self, device, dtype):
        # Convention pin: the homogeneous 1.0 is appended as the *last* component (the
        # non-symmetric rows catch a prepend or a component reversal).
        points_h = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [-1.0, -2.0, -1.0], [0.0, 1.0, -2.0]],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [1.0, 2.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 1.0],
                [2.0, 1.0, 0.0, 1.0],
                [-1.0, -2.0, -1.0, 1.0],
                [0.0, 1.0, -2.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_to_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_convert_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[2.0, 1.0, 0.0, 1.0]], [[0.0, 1.0, 2.0, 1.0]], [[0.0, 1.0, -2.0, 1.0]]], device=device, dtype=dtype
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_to_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_gradcheck(self, batch_shape, device):
        points_h = torch.rand(batch_shape, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_points_to_homogeneous, (points_h,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_points_to_homogeneous
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        self.assert_close(actual, expected)


class TestConvertAtoH(BaseTester):
    def test_convert_points(self, device, dtype):
        # Convention pin and its enforcement point: the (B, 2, 3) affine block is copied
        # verbatim into the top of the (B, 3, 3) result (no transpose, no reordering) and the
        # row [0, 0, 1] is appended at the *bottom*. The literal is non-symmetric so a
        # transpose is caught.
        # Snippet used to generate expected (torch only):
        #   convert_affinematrix_to_homography(torch.tensor([[[1., 2., 3.], [4., 5., 6.]]]))
        #     -> [[[1., 2., 3.], [4., 5., 6.], [0., 0., 1.]]]
        A = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], device=device, dtype=dtype)

        expected = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        H = kornia.geometry.conversions.convert_affinematrix_to_homography(A)
        self.assert_close(H, expected)

    @pytest.mark.parametrize("batch_shape", [(10, 2, 3), (16, 2, 3)])
    def test_gradcheck(self, batch_shape, device):
        points_h = torch.rand(batch_shape, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_affinematrix_to_homography, (points_h,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_affinematrix_to_homography
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        self.assert_close(actual, expected)

    def test_convention_homography3d_appends_bottom_row_0_0_0_1(self, device, dtype):
        # Convention pin (3-D sibling, which has no test class of its own): the (B, 3, 4) affine
        # block is copied verbatim and the row [0, 0, 0, 1] is appended at the bottom.
        # Snippet used to generate expected (by hand):
        #   [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] gains [0, 0, 0, 1]
        A = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.convert_affinematrix_to_homography3d(A)

        expected = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out, expected)


class TestConvertPointsFromHomogeneous(BaseTester):
    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_cardinality(self, device, dtype, batch_shape):
        points_h = torch.rand(batch_shape, device=device, dtype=dtype)
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        assert points.shape == points.shape[:-1] + (2,)

    def test_points(self, device, dtype):
        # Convention pins: the [2., 1., 0.] row is the |w| <= eps case (default eps 1e-8) --
        # returned *unchanged*, not zeros, not inf, no exception. The negative-w rows pin that
        # the sign of w is preserved (no abs): [0., 1., -2.] -> [0., -0.5].
        points_h = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [-1.0, -2.0, -1.0], [0.0, 1.0, -2.0]],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [[1.0, 2.0], [0.0, 0.5], [2.0, 1.0], [1.0, 2.0], [0.0, -0.5]], device=device, dtype=dtype
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor([[[2.0, 1.0]], [[0.0, 0.5]], [[0.0, -0.5]]], device=device, dtype=dtype)

        # to euclidean
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_h = torch.ones(1, 10, 3, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_points_from_homogeneous, (points_h,))

    def test_gradcheck_zvec_zeros(self, device):
        # generate input data
        points_h = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.1], [2.0, 1.0, 0.1]], device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_points_from_homogeneous, (points_h,), eps=1e-8)

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_points_from_homogeneous
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        self.assert_close(actual, expected)

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="convert_points_from_homogeneous divides by w + eps — kornia#3938",
        strict=True,
    )
    def test_convention_divides_by_exactly_w(self, device, dtype):
        # Intended behavior: for |w| > eps the point is divided by exactly w. It currently is
        # divided by w + eps with no regard for sign, so the signed relative error is exactly
        # -eps / (w + eps): -1/3 at w = +2e-8 (33 % low) and +1 at w = -2e-8 (100 % high) (#3938).
        # Marked xfail(strict=True) so fixing #3938 makes this XPASS and forces the mark out.
        # Snippet used to generate expected (by hand):
        #   2 / 2e-8, 4 / 2e-8 -> [1e8, 2e8]  (kornia returns [6.6666667e7, 1.3333333e8])
        #   measured signed relative error in float64: -0.3333333333333334, and
        #   -eps / (w + eps) = -1e-8 / 3e-8 = -0.3333333333333333
        if dtype == torch.float16:
            pytest.skip("float16 underflows w=2e-8 to 0, which is the |w| <= eps passthrough branch, not the eps bias")

        points = torch.tensor([[2.0, 4.0, 2e-8]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.convert_points_from_homogeneous(points)

        expected = torch.tensor([[1e8, 2e8]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_wart_division_is_by_w_plus_eps_for_both_signs_3938(self, device, dtype):
        # Wart pin for kornia#3938, companion to the strict xfail above: assert the CURRENT
        # biased outputs for both signs of w. The xfail pins the intended behavior but cannot
        # flip under every fix polarity -- a sign-aware eps (w + sign(w) * eps) leaves the
        # positive-w case failing the intended [1e8, 2e8] and the mark silently XFAIL; here the
        # NEGATIVE-w cell (divided by -2e-8 + 1e-8 = -1e-8, doubling the point) flips under that
        # fix too, and both cells flip under an exact division or a grad-only eps. If either
        # assert fails, #3938 was (partly) fixed -- update or remove the warning in
        # convert_points_from_homogeneous and flip/remove the strict xfail above. eps=1e-8 is
        # passed explicitly so the pinned literals do not silently track a later change to the default.
        # Snippet used to generate expected (torch only, executed at each pinned dtype):
        #   cpfh = kornia.geometry.conversions.convert_points_from_homogeneous
        #   cpfh(torch.tensor([[2., 4., 2e-8]], dtype=torch.float64), eps=1e-8)
        #     -> [[66666666.66666666, 133333333.33333331]]   (f32: [[66666668.0, 133333336.0]])
        #   cpfh(torch.tensor([[2., 4., -2e-8]], dtype=torch.float64), eps=1e-8)
        #     -> [[-200000000.0, -400000000.0]]               (f32: identical)
        # At bfloat16 the outputs land within 0.15 % of the literals (66584576, -200278016),
        # inside rtol 1e-2, so the pin holds there too.
        if dtype == torch.float16:
            pytest.skip("float16 underflows w=2e-8 to 0, which is the |w| <= eps passthrough branch, not the eps bias")

        cpfh = kornia.geometry.conversions.convert_points_from_homogeneous

        out_pos = cpfh(torch.tensor([[2.0, 4.0, 2e-8]], device=device, dtype=dtype), eps=1e-8)
        out_neg = cpfh(torch.tensor([[2.0, 4.0, -2e-8]], device=device, dtype=dtype), eps=1e-8)

        expected_pos = torch.tensor([[6.6666668e7, 1.33333336e8]], device=device, dtype=dtype)
        expected_neg = torch.tensor([[-2e8, -4e8]], device=device, dtype=dtype)
        self.assert_close(out_pos, expected_pos, atol=0.0, rtol=1e-2)
        self.assert_close(out_neg, expected_neg, atol=0.0, rtol=1e-2)


def _skip_if_mps_clamp_caching(device):
    # Runtime probe instead of a torch-version pin, so the skip retires itself on any torch
    # build where the two clamps below return different values.
    if device.type == "mps" and torch.equal(
        torch.zeros(2, device=device).clamp(1e-8), torch.zeros(2, device=device).clamp(1e-7)
    ):
        pytest.skip(
            "this torch build caches clamp's scalar min per shape/dtype on MPS -- first value wins "
            "(seen on torch 2.9.1): z = torch.zeros(2, device='mps'); z.clamp(1e-8) then z.clamp(1e-7) "
            "both return 9.99999993922529e-09, while the same pair on cpu returns 1e-08 then "
            "1.0000000116860974e-07. The clamped eps this pin measures is therefore set by whichever "
            "earlier test clamped first, which is a torch defect, not a kornia one"
        )


def _assert_degenerate_size_cell(
    func_2d, func_3d, fill, ndim, arg_name, degenerate_size, expected, tols, device, dtype
):
    # Shared driver for the two kornia#3940 wart matrices below -- the normalize and
    # denormalize halves differ only in function pair, fill value, tolerances and expected
    # table -- so the eventual #3940 cleanup edits one body and one MPS-skip helper. eps=1e-8 is
    # passed explicitly so the pinned literals do not silently track a later change to the default eps
    # while the clamp bug itself is still present.
    _skip_if_mps_clamp_caching(device)

    if ndim == "2d":
        sizes = {"height": 5, "width": 7}
        func = func_2d
    else:
        sizes = {"depth": 5, "height": 7, "width": 9}
        func = func_3d
    sizes[arg_name] = degenerate_size
    pts = torch.full((1, len(sizes)), fill, device=device, dtype=dtype)

    out = func(pts, *sizes.values(), eps=1e-8)

    expected_t = torch.tensor([expected], device=device, dtype=dtype)
    if tols is None:
        assert_close(out, expected_t)
    else:
        atol, rtol = tols
        assert_close(out, expected_t, atol=atol, rtol=rtol)


class TestNormalizePixelCoordinates(BaseTester):
    def test_tensor_bhw2(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.geometry.conversions.normalize_pixel_coordinates(grid, height, width, eps=eps)

        self.assert_close(grid_norm, expected, atol=atol, rtol=rtol)

    def test_list(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.geometry.conversions.normalize_pixel_coordinates(grid, height, width, eps=eps)

        self.assert_close(grid_norm, expected, atol=atol, rtol=rtol)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if device == torch.device("cpu"):
            pytest.skip("NormalizePixelCoordinates not working on CPU with dynamo!")

        op = kornia.geometry.conversions.normalize_pixel_coordinates
        op_optimized = torch_optimizer(op)

        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        actual = op_optimized(grid, height, width)
        expected = op(grid, height, width)

        self.assert_close(actual, expected)

    def test_convention_corner_aligned_formula(self, device, dtype):
        # Convention pin: normalize_pixel_coordinates maps x -> 2*x/(W - 1) - 1 (corner-aligned,
        # i.e. the align_corners=True convention). grid_sample's *default* align_corners=False
        # convention, (2*x + 1)/W - 1, would give [-0.75, -0.25, 0.75] for the same input.
        # Snippet used to generate expected (stdlib only, W = 4):
        #   [2 * x / (4 - 1) - 1 for x in (0.0, 1.0, 3.0)] -> [-1.0, -0.3333333333333333, 1.0]
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 4, 4)

        expected = torch.tensor([[-1.0, -1.0], [-0.33333333, -0.33333333], [1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_positional_order_is_height_then_width(self, device, dtype):
        # Convention pin: the positional signature is (pixel_coordinates, height, width), which is
        # the reverse of the per-point (x, y) -> (width, height) scaling order: slot 0 is scaled by
        # width and slot 1 by height. Calling with H and W swapped would give [[5.0, -0.3333]].
        # Snippet used to generate expected (stdlib only, H = 2, W = 4):
        #   2 * 3.0 / (4 - 1) - 1, 2 * 1.0 / (2 - 1) - 1 -> (1.0, 1.0)
        pts = torch.tensor([[3.0, 1.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 2, 4)

        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_output_is_not_clamped(self, device, dtype):
        # Convention pin: nothing is clamped to [-1, 1] -- out-of-image coordinates extrapolate
        # linearly past it.
        # Snippet used to generate expected (stdlib only, H = W = 4):
        #   2 * 10.0 / (4 - 1) - 1, 2 * 0.0 / (4 - 1) - 1 -> (5.666666666666666, -1.0)
        pts = torch.tensor([[10.0, 0.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 4, 4)

        expected = torch.tensor([[5.6666667, -1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_grid_sample_needs_align_corners_true(self, device, dtype):
        # Convention pin: feeding normalized coordinates to torch.nn.functional.grid_sample
        # requires align_corners=True to be passed explicitly. With it, the three normalized
        # pixel centres sample back the exact pixel values; grid_sample's own default
        # (align_corners=None -> False) instead places u = -1, -1/3, 1 at pixels
        # ((u + 1) * 4 - 1) / 2 = -0.5, 0.8333, 3.5, i.e. half a pixel outside the image at
        # both ends, so every sampled value would be wrong.
        # Snippet used to generate expected (stdlib only, W = 4, img = arange(16).view(4, 4)):
        #   img[0, 0], img[1, 1], img[3, 3] -> 0.0, 5.0, 15.0
        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], device=device, dtype=dtype)
        grid = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 4, 4).view(1, 1, 3, 2)

        sampled_aligned = torch.nn.functional.grid_sample(img, grid, align_corners=True).flatten()

        self.assert_close(sampled_aligned, torch.tensor([0.0, 5.0, 15.0], device=device, dtype=dtype))

    def test_convention_3d_component_order_is_depth_x_y(self, device, dtype):
        # Convention pin (normalize_pixel_coordinates3d has no test class of its own): the
        # component order is (d, x, y) -- depth first, then x scaled by width, then y scaled by
        # height. It is NOT (x, y, z): reading the same three numbers that way sends the point
        # out of range instead of to the far corner.
        # Snippet used to generate expected (stdlib only, D = 3, H = 5, W = 9):
        #   2 * 2 / (3 - 1) - 1, 2 * 8 / (9 - 1) - 1, 2 * 4 / (5 - 1) - 1 -> (1.0, 1.0, 1.0)
        #   the (x, y, z) reading [2, 4, 8] gives 2 * 2 / 2 - 1, 2 * 4 / 8 - 1, 2 * 8 / 4 - 1
        #                                      -> (1.0, 0.0, 3.0)
        far_corner = torch.tensor([[2.0, 8.0, 4.0]], device=device, dtype=dtype)
        out = kornia.geometry.conversions.normalize_pixel_coordinates3d(far_corner, 3, 5, 9)
        self.assert_close(out, torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=dtype))

        swapped = torch.tensor([[2.0, 4.0, 8.0]], device=device, dtype=dtype)
        out_swapped = kornia.geometry.conversions.normalize_pixel_coordinates3d(swapped, 3, 5, 9)
        self.assert_close(out_swapped, torch.tensor([[1.0, 0.0, 3.0]], device=device, dtype=dtype))

    # Wart-pin matrix for kornia#3940, normalizing half: one cell per (size argument x
    # degenerate class) of normalize_pixel_coordinates and normalize_pixel_coordinates3d,
    # asserting the CURRENT broken output that the docstring warnings document. If any cell
    # fails, #3940 was (partly) fixed -- update or remove the degenerate-size warnings in
    # normalize_pixel_coordinates, denormalize_pixel_coordinates, normalize_pixel_coordinates3d
    # and denormalize_pixel_coordinates3d. The cells are NOT a contract that degenerate sizes
    # must keep returning these values.
    # They are regular tests rather than strict xfails on purpose: the intended behavior
    # (raise ValueError, or clamp the *output*, or keep the current pass-through) is a
    # maintainer decision, and a strict xfail asserting one of those answers would stay
    # silently XFAIL forever if a different one were chosen. A wart pin flips loudly under
    # every polarity, and covering the full matrix means any partial fix -- one function, one
    # argument, or one degenerate class -- flips at least one cell.
    # Exactly one size argument is degenerate per cell (the others stay at 5/7 in 2-D and
    # 5/7/9 in 3-D), so the finite components pin which argument was degenerated. All three
    # classes give the same output because the mechanism is `(size - 1).clamp(eps)`: size 1
    # gives 0, size 0 gives -1 and size -3 gives -4, all clamped up to eps = 1e-8, so the
    # factor becomes 2e8.
    # Snippet used to generate expected (torch only; same output for bad in (1, 0, -3)):
    #   npc = kornia.geometry.conversions.normalize_pixel_coordinates
    #   npc3 = kornia.geometry.conversions.normalize_pixel_coordinates3d
    #   npc(torch.tensor([[1., 1.]], dtype=torch.float64), bad, 7, eps=1e-8)
    #     -> [[-0.6666666666666667, 199999999.0]]
    #   npc(torch.tensor([[1., 1.]], dtype=torch.float64), 5, bad, eps=1e-8) -> [[199999999.0, -0.5]]
    #   npc3(torch.tensor([[1., 1., 1.]], dtype=torch.float64), bad, 7, 9, eps=1e-8)
    #     -> [[199999999.0, -0.75, -0.6666666666666667]]
    #   npc3(torch.tensor([[1., 1., 1.]], dtype=torch.float64), 5, bad, 9, eps=1e-8)
    #     -> [[-0.5, -0.75, 199999999.0]]
    #   npc3(torch.tensor([[1., 1., 1.]], dtype=torch.float64), 5, 7, bad, eps=1e-8)
    #     -> [[-0.5, 199999999.0, -0.6666666666666667]]
    # At float16 2e8 overflows to inf and the literal overflows identically; at bfloat16 both
    # sides round to 200278016.0, so the comparison stays meaningful at every dtype.
    @pytest.mark.parametrize("degenerate_size", [1, 0, -3], ids=["one", "zero", "negative"])
    @pytest.mark.parametrize(
        ("ndim", "arg_name", "expected"),
        [
            ("2d", "height", [-0.6666667, 199999999.0]),
            ("2d", "width", [199999999.0, -0.5]),
            ("3d", "depth", [199999999.0, -0.75, -0.6666667]),
            ("3d", "height", [-0.5, -0.75, 199999999.0]),
            ("3d", "width", [-0.5, 199999999.0, -0.6666667]),
        ],
        ids=["2d-height", "2d-width", "3d-depth", "3d-height", "3d-width"],
    )
    def test_wart_degenerate_size_matrix(self, ndim, arg_name, degenerate_size, expected, device, dtype):
        _assert_degenerate_size_cell(
            kornia.geometry.conversions.normalize_pixel_coordinates,
            kornia.geometry.conversions.normalize_pixel_coordinates3d,
            1.0,
            ndim,
            arg_name,
            degenerate_size,
            expected,
            None,
            device,
            dtype,
        )

    def test_wart_all_size_arguments_degenerate_together(self, device, dtype):
        # Wart pin for kornia#3940, companion to the matrix above: both sizes degenerate at
        # once also passes silently, exploding every component. Flips together with the matrix
        # when #3940 is fixed -- see the cleanup note above the matrix.
        # Snippet used to generate expected (torch only):
        #   npc = kornia.geometry.conversions.normalize_pixel_coordinates
        #   npc(torch.tensor([[1., 1.]], dtype=torch.float64), 1, 1, eps=1e-8)
        #     -> [[199999999.0, 199999999.0]]
        _skip_if_mps_clamp_caching(device)

        pts = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 1, 1, eps=1e-8)

        expected = torch.tensor([[199999999.0, 199999999.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)


def test_wart_default_eps_1e_8_backs_the_quoted_warning_numbers():
    # The wart pins in this file pass eps=1e-8 explicitly so their literals do not track the
    # default, which leaves the default itself pinned by nothing while six docstring warnings
    # quote numbers that hold only for eps=1e-8: cart2pol (rho = 1e-04 at the origin),
    # convert_points_from_homogeneous (the -1/3 and +1 relative errors at w = +/-2e-8),
    # normalize_pixel_coordinates and normalize_pixel_coordinates3d (the 199999999.0 / 2e8
    # blow-up factor) and denormalize_pixel_coordinates / denormalize_pixel_coordinates3d (the
    # 5e-09 collapse factor). If this fails, the default moved -- rework those warnings'
    # numbers together with this list.
    for op_name in (
        "cart2pol",
        "convert_points_from_homogeneous",
        "normalize_pixel_coordinates",
        "denormalize_pixel_coordinates",
        "normalize_pixel_coordinates3d",
        "denormalize_pixel_coordinates3d",
    ):
        op = getattr(kornia.geometry.conversions, op_name)
        assert inspect.signature(op).parameters["eps"].default == 1e-8, op_name


def test_wart_float16_underflowed_default_eps_flips_branches(device):
    # Wart pins for the float16 sentences of the #3939 and #3938 warnings. float16 is
    # hardcoded (no dtype fixture) so the pins run in every test configuration: the float16
    # legs of the wart pins above are skipped because the default eps=1e-8 underflows to 0
    # there, which is exactly the behavior pinned here. eps is left at its default on purpose
    # -- the underflow of the *default* is the claim. atol=rtol=0.0 because both claims are
    # exactness claims: with the float16 default tolerance (1e-3) the eps-biased
    # rho = 1e-4 of the other branch would still compare equal to 0.
    # Snippet used to generate expected (torch only, executed on cpu float16):
    #   cart2pol(torch.tensor(0., dtype=torch.float16), torch.tensor(0., dtype=torch.float16))[0]
    #     -> 0.0  (not sqrt(eps) = 1e-4: eps underflows the sum inside the sqrt)
    #   convert_points_from_homogeneous(torch.tensor([[2., 4., 2e-8]], dtype=torch.float16))
    #     -> [[2., 4.]]  (w underflows to 0 and takes the abs(w) <= eps passthrough branch)
    zero = torch.tensor(0.0, device=device, dtype=torch.float16)
    rho = kornia.geometry.conversions.cart2pol(zero, zero)[0]
    assert_close(rho, zero, atol=0.0, rtol=0.0)

    out = kornia.geometry.conversions.convert_points_from_homogeneous(
        torch.tensor([[2.0, 4.0, 2e-8]], device=device, dtype=torch.float16)
    )
    assert_close(out, torch.tensor([[2.0, 4.0]], device=device, dtype=torch.float16), atol=0.0, rtol=0.0)


def test_wart_float16_degenerate_roundtrip_is_inf_then_nan(device):
    # Wart pin for the float16 sentence of the #3940 warnings, float16-hardcoded like the test
    # above: with the default eps underflowed to 0, the clamp keeps the degenerate denominator
    # at 0, the normalized component is inf (not the 2e8 the other dtypes pin) and the
    # denormalize round trip of that is nan, not the input.
    # Snippet used to generate expected (torch only, executed on cpu float16):
    #   normalize_pixel_coordinates(torch.ones(1, 2, dtype=torch.float16), 1, 1) -> [[inf, inf]]
    #   denormalize_pixel_coordinates(<that>, 1, 1) -> [[nan, nan]]
    _skip_if_mps_clamp_caching(device)

    ones = torch.ones(1, 2, device=device, dtype=torch.float16)
    norm = kornia.geometry.conversions.normalize_pixel_coordinates(ones, 1, 1)
    assert (norm == torch.inf).all()

    denorm = kornia.geometry.conversions.denormalize_pixel_coordinates(norm, 1, 1)
    assert torch.isnan(denorm).all()


class TestDenormalizePixelCoordinates(BaseTester):
    def test_tensor_bhw2(self, device, dtype):
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.geometry.conversions.denormalize_pixel_coordinates(grid, height, width)

        self.assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_list(self, device, dtype):
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.geometry.conversions.denormalize_pixel_coordinates(grid, height, width)

        self.assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if device == torch.device("cpu"):
            pytest.xfail("DenormalizePixelCoordinates not working on CPU with dynamo!")

        op = kornia.geometry.conversions.denormalize_pixel_coordinates
        op_optimized = torch_optimizer(op)

        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        actual = op_optimized(grid, height, width)
        expected = op(grid, height, width)

        self.assert_close(actual, expected)

    def test_convention_corner_aligned_inverse(self, device, dtype):
        # Convention pin: denormalize_pixel_coordinates is the corner-aligned inverse,
        # x = (W - 1) * (x_norm + 1) / 2, taken positionally as (coords, height, width) with
        # (x, y) points. grid_sample's align_corners=False convention, ((x_norm + 1) * W - 1)/2,
        # would give [[3.5, -0.5]] for the same input.
        # Snippet used to generate expected (stdlib only, H = 2, W = 4):
        #   (4 - 1) * (1.0 + 1) / 2, (2 - 1) * (-1.0 + 1) / 2 -> (3.0, 0.0)
        pts_norm = torch.tensor([[1.0, -1.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.denormalize_pixel_coordinates(pts_norm, 2, 4)

        expected = torch.tensor([[3.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_roundtrip_denormalize_of_normalize(self, device, dtype):
        # Convention pin: denormalize(normalize(p)) == p on a non-degenerate, non-square,
        # non-identity image size, so the two formulas are exact mutual inverses.
        # Snippet used to generate expected (by hand): the input itself.
        pts = torch.tensor([[1.0, 2.0], [3.0, 0.0]], device=device, dtype=dtype)

        norm = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 5, 7)
        out = kornia.geometry.conversions.denormalize_pixel_coordinates(norm, 5, 7)

        self.assert_close(out, pts)

    def test_convention_3d_component_order_and_roundtrip(self, device, dtype):
        # Convention pin (denormalize_pixel_coordinates3d has no test class of its own): same
        # (d, x, y) order as the 3-D normalizer, so the normalized origin maps to the per-axis
        # centres ((D - 1)/2, (W - 1)/2, (H - 1)/2), and the pair round-trips exactly.
        # Snippet used to generate expected (stdlib only, D = 3, H = 5, W = 9):
        #   (3 - 1) / 2, (9 - 1) / 2, (5 - 1) / 2 -> (1.0, 4.0, 2.0)
        centre = kornia.geometry.conversions.denormalize_pixel_coordinates3d(
            torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype), 3, 5, 9
        )
        self.assert_close(centre, torch.tensor([[1.0, 4.0, 2.0]], device=device, dtype=dtype))

        pts = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        norm = kornia.geometry.conversions.normalize_pixel_coordinates3d(pts, 3, 5, 9)
        out = kornia.geometry.conversions.denormalize_pixel_coordinates3d(norm, 3, 5, 9)
        self.assert_close(out, pts)

    # Wart-pin matrix for kornia#3940, denormalizing half: one cell per (size argument x
    # degenerate class) of denormalize_pixel_coordinates and denormalize_pixel_coordinates3d,
    # asserting the CURRENT broken output that the docstring warnings document. If any cell
    # fails, #3940 was (partly) fixed -- update or remove the degenerate-size warnings in
    # normalize_pixel_coordinates, denormalize_pixel_coordinates, normalize_pixel_coordinates3d
    # and denormalize_pixel_coordinates3d. The cells are NOT a contract that degenerate sizes
    # must keep returning these values; see the polarity note on the normalize wart matrix,
    # which also explains why exactly one argument is degenerate per cell and why all three
    # classes (1, 0, -3) give the same output.
    # Here the clamped denominator multiplies instead of divides, so the degenerate axis
    # collapses to eps / 2 = 5e-09 rather than exploding to 2e8. The tolerance is tight
    # (rtol 1e-6, atol 0) on purpose: at atol 1e-2 the collapsed component would compare
    # equal to any small number, including the 0.0 a "clamp the output" fix might return.
    # It still holds at every dtype -- at float16 5e-09 underflows to 0.0 on both the
    # measured and the literal side, at bfloat16 both round to 5.005858838558197e-09, and the
    # finite components 2.0/3.0/4.0 are exact in every dtype.
    # Snippet used to generate expected (torch only; same output for bad in (1, 0, -3)):
    #   dpc = kornia.geometry.conversions.denormalize_pixel_coordinates
    #   dpc3 = kornia.geometry.conversions.denormalize_pixel_coordinates3d
    #   dpc(torch.tensor([[0., 0.]], dtype=torch.float64), bad, 7, eps=1e-8) -> [[3.0, 5e-09]]
    #   dpc(torch.tensor([[0., 0.]], dtype=torch.float64), 5, bad, eps=1e-8) -> [[5e-09, 2.0]]
    #   dpc3(torch.tensor([[0., 0., 0.]], dtype=torch.float64), bad, 7, 9, eps=1e-8)
    #     -> [[5e-09, 4.0, 3.0]]
    #   dpc3(torch.tensor([[0., 0., 0.]], dtype=torch.float64), 5, bad, 9, eps=1e-8)
    #     -> [[2.0, 4.0, 5e-09]]
    #   dpc3(torch.tensor([[0., 0., 0.]], dtype=torch.float64), 5, 7, bad, eps=1e-8)
    #     -> [[2.0, 5e-09, 3.0]]
    @pytest.mark.parametrize("degenerate_size", [1, 0, -3], ids=["one", "zero", "negative"])
    @pytest.mark.parametrize(
        ("ndim", "arg_name", "expected"),
        [
            ("2d", "height", [3.0, 5e-09]),
            ("2d", "width", [5e-09, 2.0]),
            ("3d", "depth", [5e-09, 4.0, 3.0]),
            ("3d", "height", [2.0, 4.0, 5e-09]),
            ("3d", "width", [2.0, 5e-09, 3.0]),
        ],
        ids=["2d-height", "2d-width", "3d-depth", "3d-height", "3d-width"],
    )
    def test_wart_degenerate_size_matrix(self, ndim, arg_name, degenerate_size, expected, device, dtype):
        _assert_degenerate_size_cell(
            kornia.geometry.conversions.denormalize_pixel_coordinates,
            kornia.geometry.conversions.denormalize_pixel_coordinates3d,
            0.0,
            ndim,
            arg_name,
            degenerate_size,
            expected,
            (0.0, 1e-6),
            device,
            dtype,
        )


class TestProjectPoints(BaseTester):
    def test_smoke(self, device, dtype):
        point_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        point_3d = torch.zeros(2, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 2)

    def test_smoke_batch_multi(self, device, dtype):
        point_3d = torch.zeros(2, 4, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, 4, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 4, 2)

    def test_project_and_unproject(self, device, dtype):
        point_3d = torch.tensor([[10.0, 2.0, 30.0]], device=device, dtype=dtype)
        depth = point_3d[..., -1:]
        camera_matrix = torch.tensor(
            [[[2746.0, 0.0, 991.0], [0.0, 2748.0, 619.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        point_3d_hat = kornia.geometry.camera.unproject_points(point_2d, depth, camera_matrix)
        self.assert_close(point_3d, point_3d_hat, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # TODO: point [0, 0, 0] crashes
        points_3d = torch.ones(1, 3, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.camera.project_points, (points_3d, camera_matrix))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.camera.project_points
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_3d, camera_matrix)
        expected = op(points_3d, camera_matrix)

        self.assert_close(actual, expected)


class TestDenormalizePointsWithIntrinsics(BaseTester):
    def test_smoke(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        points_norm = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        points_2d = torch.zeros(2, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 2)

    def test_smoke_batch_n(self, device, dtype):
        points_2d = torch.zeros(2, 9, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 9, 2)

    def test_toy(self, device, dtype):
        point_2d = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.denormalize_points_with_intrinsics
        expected = torch.tensor([[192.0, 192.0]], device=device, dtype=dtype)
        self.assert_close(op(point_2d, camera_matrix), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_2d = torch.zeros(1, 2, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.denormalize_points_with_intrinsics, (points_2d, camera_matrix))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.conversions.denormalize_points_with_intrinsics
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_2d, camera_matrix)
        expected = op(points_2d, camera_matrix)

        self.assert_close(actual, expected)

    def test_convention_maps_normalized_camera_points_to_pixels(self, device, dtype):
        # Convention pin: the input is in *normalized camera* coordinates and the output is in
        # *pixel* coordinates, using the pinhole layout u = x * fx + cx, v = y * fy + cy.
        # fx != fy and cx != cy so a transposed or swapped read of K is caught.
        # Snippet used to generate expected (stdlib only, fx=100, fy=200, cx=320, cy=240):
        #   1.0 * 100 + 320, 1.0 * 200 + 240 -> (420.0, 440.0)
        points_norm = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_norm, camera_matrix)

        expected = torch.tensor([[420.0, 440.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)


class TestNormalizePointsWithIntrinsics(BaseTester):
    def test_smoke(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        points_norm = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        points_2d = torch.zeros(2, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 2)

    def test_smoke_batch_n(self, device, dtype):
        points_2d = torch.zeros(2, 10, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 10, 2)

    def test_norm_unnorm(self, device, dtype):
        point_2d = torch.tensor([[128.0, 128.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        back = kornia.geometry.conversions.denormalize_points_with_intrinsics
        point_2d_norm = op(point_2d, camera_matrix)
        point_2d_hat = back(point_2d_norm, camera_matrix)
        self.assert_close(point_2d, point_2d_hat, atol=1e-4, rtol=1e-4)

    def test_toy(self, device, dtype):
        point_2d = torch.tensor([[192.0, 192.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        out = op(point_2d, camera_matrix)
        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_2d = torch.zeros(1, 2, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.normalize_points_with_intrinsics, (points_2d, camera_matrix))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_2d, camera_matrix)
        expected = op(points_2d, camera_matrix)

        self.assert_close(actual, expected)

    def test_convention_intrinsics_layout_fx_fy_cx_cy(self, device, dtype):
        # Convention pin: K is the standard row-major pinhole matrix, fx = K[..., 0, 0],
        # fy = K[..., 1, 1], cx = K[..., 0, 2], cy = K[..., 1, 2], and the point is (u, v) in
        # pixels, so x = (u - cx)/fx, y = (v - cy)/fy. fx != fy and cx != cy, so swapping the
        # two focal lengths would give [[0.5, 2.0]] and a transposed K would not divide at all.
        # Snippet used to generate expected (stdlib only, fx=100, fy=200, cx=320, cy=240):
        #   (420.0 - 320) / 100, (440.0 - 240) / 200 -> (1.0, 1.0)
        points_2d = torch.tensor([[420.0, 440.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)

        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_skew_term_is_ignored(self, device, dtype):
        # Convention pin: only the diagonal fx, fy and the [:2, 2] column of K are read -- the
        # skew entry K[..., 0, 1] is silently ignored, so a skewed K gives the same answer as
        # the skew-free one. A skew-aware implementation would return (1.0 - 7/100, 1.0).
        # Snippet used to generate expected (stdlib only): identical to the skew-free result,
        #   (420.0 - 320) / 100, (440.0 - 240) / 200 -> (1.0, 1.0)
        points_2d = torch.tensor([[420.0, 440.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 7.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)

        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)


class TestRt2Extrinsics(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        R = torch.rand(batch_size, 3, 3, dtype=dtype, device=device)
        t = torch.rand(batch_size, 3, 1, dtype=dtype, device=device)

        Rt = Rt_to_matrix4x4(R, t)
        assert Rt.shape == (batch_size, 4, 4)

        R2, t2 = matrix4x4_to_Rt(Rt)
        assert R2.shape == (batch_size, 3, 3)
        assert t2.shape == (batch_size, 3, 1)

        self.assert_close(R, R2, rtol=1e-4, atol=1e-5)
        self.assert_close(t, t2, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [5])
    def test_gradcheck(self, batch_size, device):
        R = torch.rand(batch_size, 3, 3, dtype=torch.float64, device=device)
        t = torch.rand(batch_size, 3, 1, dtype=torch.float64, device=device)
        self.gradcheck(kornia.geometry.conversions.Rt_to_matrix4x4, (R, t))


class TestCamtoworldGraphicsToVision(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        t_vis = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R_vis = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        K_vis = Rt_to_matrix4x4(R_vis, t_vis)
        K_graf = camtoworld_vision_to_graphics_4x4(K_vis)

        expected = torch.tensor(
            [[0, 0, -1, 2], [0, -1, 0, 3], [-1, 0, 0, 4], [0, 0, 0, 1]], device=device, dtype=dtype
        )[None].repeat(batch_size, 1, 1)

        self.assert_close(K_graf, expected, rtol=1e-4, atol=1e-5)
        R_graf, t_graf = camtoworld_vision_to_graphics_Rt(R_vis, t_vis)
        expected_R = torch.tensor([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], device=device, dtype=dtype)[None].repeat(
            batch_size, 1, 1
        )
        expected_t = torch.tensor([2, 3, 4], device=device, dtype=dtype).reshape(1, 3, 1).repeat(batch_size, 1, 1)

        self.assert_close(t_graf, expected_t, rtol=1e-4, atol=1e-5)
        self.assert_close(R_graf, expected_R, rtol=1e-4, atol=1e-5)

        Kvis_back = camtoworld_graphics_to_vision_4x4(K_graf)
        self.assert_close(Kvis_back, K_vis, rtol=1e-4, atol=1e-5)

        R_vis_back, t_vis_back = camtoworld_graphics_to_vision_Rt(R_graf, t_graf)
        self.assert_close(R_vis_back, R_vis, rtol=1e-4, atol=1e-5)
        self.assert_close(t_vis_back, t_vis, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device):
        t_vis = torch.tensor([2, 3, 4], device=device, dtype=torch.float64).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=torch.float64)[None]
        R_vis = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        K_vis = Rt_to_matrix4x4(R_vis, t_vis)
        self.gradcheck(camtoworld_graphics_to_vision_4x4, (K_vis,))
        self.gradcheck(camtoworld_vision_to_graphics_4x4, (K_vis,))


class TestCamtoworldRtToPoseRt(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        t = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)

        Rp, tp = camtoworld_to_worldtocam_Rt(R, t)

        expected_Rp = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=device, dtype=dtype)[None].repeat(
            batch_size, 1, 1
        )
        expected_tp = torch.tensor([4, -3, -2], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        self.assert_close(Rp, expected_Rp, rtol=1e-4, atol=1e-5)
        self.assert_close(tp, expected_tp, rtol=1e-4, atol=1e-5)

        Rback, tback = worldtocam_to_camtoworld_Rt(Rp, tp)
        self.assert_close(Rback, R, rtol=1e-4, atol=1e-5)
        self.assert_close(tback, t, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device):
        t = torch.tensor([2, 3, 4], device=device, dtype=torch.float64).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=torch.float64)[None]
        R = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        self.gradcheck(camtoworld_to_worldtocam_Rt, (R, t))
        self.gradcheck(worldtocam_to_camtoworld_Rt, (R, t))


class TestCARKitToColmap(BaseTester):
    def test_everything(self, device, dtype):
        # generate input data
        t = torch.tensor([1, 0, 0], device=device, dtype=dtype).view(1, 3, 1)
        ang_deg = torch.tensor([45, 60.0, 0.0], device=device, dtype=dtype)[None]
        ang_rad = kornia.geometry.conversions.deg2rad(ang_deg)
        qvec = kornia.geometry.axis_angle_to_quaternion(ang_rad)

        q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(qvec, t)

        angles_colmap = kornia.geometry.conversions.quaternion_to_axis_angle(q_colmap)
        angles_colmap = kornia.geometry.conversions.rad2deg(angles_colmap)
        expected_angles = torch.tensor([[116.8870620728, 0.0, -71.7524719238]], device=device, dtype=dtype)
        expected_t = torch.tensor([[[-0.5256], [0.3558], [0.7727]]], device=device, dtype=dtype)

        self.assert_close(angles_colmap, expected_angles, rtol=1e-4, atol=1e-5)
        self.assert_close(t_colmap, expected_t, rtol=1e-4, atol=1e-5)


class TestEulerFromQuaternion(BaseTester):
    def test_smoke(self, device, dtype):
        q = Quaternion.random(batch_size=1)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        assert roll.shape == pitch.shape
        assert pitch.shape == yaw.shape

    @pytest.mark.parametrize("batch_size", ((1, 3, 4)))
    def test_cardinality(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size=batch_size)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        assert roll.shape[0] == batch_size
        assert pitch.shape[0] == batch_size
        assert yaw.shape[0] == batch_size

    def test_exception(self, device, dtype):
        q = Quaternion.random(batch_size=2)
        q = q.to(device, dtype)
        with pytest.raises(Exception):
            euler_from_quaternion(q.w, torch.rand(1), q.y, q.z)

    def test_gradcheck(self, device):
        q = Quaternion.random(batch_size=1).to(device, torch.float64)
        self.gradcheck(euler_from_quaternion, (q.w, q.x, q.y, q.z))

    @pytest.mark.skipif(
        torch_version() in {"2.0.1", "2.1.2", "2.2.2", "2.3.1"} and sys.version_info.minor == 8,
        reason="Not working on 2.0",
    )
    def test_dynamo(self, device, dtype, torch_optimizer):
        q = Quaternion.random(batch_size=1)
        q = q.to(device, dtype)
        op = euler_from_quaternion
        op_optimized = torch_optimizer(op)
        self.assert_close(op(q.w, q.x, q.y, q.z), op_optimized(q.w, q.x, q.y, q.z))

    def test_forth_and_back(self, device, dtype):
        q = Quaternion.random(batch_size=2)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        # TODO: check hwo to prevent getting inverted angles sometimes
        self.assert_close(q.w.abs(), qw.abs())
        self.assert_close(q.x.abs(), qx.abs())
        self.assert_close(q.y.abs(), qy.abs())
        self.assert_close(q.z.abs(), qz.abs())

    def test_convention_roll_is_x_pitch_is_y_yaw_is_z(self, device, dtype):
        # Convention pin: the three returned angles are (roll, pitch, yaw) in that order, and they
        # are rotations about x, y and z respectively -- a rotation about a single axis puts its
        # angle in exactly one slot and leaves the other two at zero, which no permutation of the
        # naming could reproduce. The return is a TUPLE of three separate tensors, not a stacked
        # (*, 3) tensor, so it cannot be indexed or sliced like one; that is pinned first.
        # The angle is 0.6 rad rather than a quarter turn so the pin stays far from the pitch =
        # +-pi/2 gimbal lock where this function does not recover the input at all.
        # Snippet used to generate the inputs (stdlib only):
        #   import math
        #   for each axis: q = (cos(0.3), sin(0.3) * axis) with 0.3 = theta / 2
        #     cos(0.3), sin(0.3) -> (0.955336489125606, 0.29552020666133955)
        w = torch.tensor(0.955336489125606, device=device, dtype=dtype)
        s = torch.tensor(0.29552020666133955, device=device, dtype=dtype)
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        expected_angle = torch.tensor(0.6, device=device, dtype=dtype)

        about_x = euler_from_quaternion(w, s, zero, zero)
        assert isinstance(about_x, tuple)
        assert len(about_x) == 3
        self.assert_close(about_x[0], expected_angle)
        self.assert_close(about_x[1], zero)
        self.assert_close(about_x[2], zero)

        about_y = euler_from_quaternion(w, zero, s, zero)
        self.assert_close(about_y[0], zero)
        self.assert_close(about_y[1], expected_angle)
        self.assert_close(about_y[2], zero)

        about_z = euler_from_quaternion(w, zero, zero, s)
        self.assert_close(about_z[0], zero)
        self.assert_close(about_z[1], zero)
        self.assert_close(about_z[2], expected_angle)

    def test_convention_euler_and_quaternion_are_mutual_inverses(self, device, dtype):
        # Convention pin: away from gimbal lock, euler_from_quaternion and quaternion_from_euler
        # invert each other exactly -- the same three angles come back, with their signs, and so
        # do the same four quaternion coefficients. Pinned at |pitch| = 0.7 < pi/4-ish and three
        # distinct non-symmetric angles so neither a permutation nor a sign flip survives. (At
        # pitch = +-pi/2 the pair is NOT a mutual inverse; that failure is out of scope here.)
        # Snippet used to generate expected (stdlib only):
        #   the round-trip is the identity on (roll, pitch, yaw) = (0.3, 0.7, 1.1)
        #   quaternion_from_euler(0.3, 0.7, 1.1) at float64 ->
        #     [0.8186292656554958, -0.057539988180335386, 0.3624200943552256, 0.44179967222724353]
        #   which is qz (x) qy (x) qx with qa = (cos(a/2), sin(a/2) * axis) -- see
        #   TestQuaternionFromEuler.test_convention_composition_is_rz_ry_rx
        roll = torch.tensor(0.3, device=device, dtype=dtype)
        pitch = torch.tensor(0.7, device=device, dtype=dtype)
        yaw = torch.tensor(1.1, device=device, dtype=dtype)

        quaternion = quaternion_from_euler(roll, pitch, yaw)
        roll_back, pitch_back, yaw_back = euler_from_quaternion(*quaternion)

        self.assert_close(roll_back, roll)
        self.assert_close(pitch_back, pitch)
        self.assert_close(yaw_back, yaw)

        quaternion_back = quaternion_from_euler(roll_back, pitch_back, yaw_back)
        for component, component_back in zip(quaternion, quaternion_back):
            self.assert_close(component_back, component)


class TestQuaternionFromEuler(BaseTester):
    def test_smoke(self, device, dtype):
        roll, pitch, yaw = torch.rand(3, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        assert qw.shape == qx.shape
        assert qx.shape == qy.shape
        assert qy.shape == qz.shape

    @pytest.mark.parametrize("batch_size", ((1, 3, 4)))
    def test_cardinality(self, device, dtype, batch_size):
        roll, pitch, yaw = torch.rand(3, batch_size, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        assert qw.shape[0] == batch_size
        assert qx.shape[0] == batch_size
        assert qy.shape[0] == batch_size
        assert qz.shape[0] == batch_size

    def test_exception(self, device, dtype):
        _, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)
        with pytest.raises(Exception):
            quaternion_from_euler(torch.rand(1), pitch, yaw)

    def test_gradcheck(self, device):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(quaternion_from_euler, (roll, pitch, yaw))

    def test_dynamo(self, device, dtype, torch_optimizer):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)

        op = quaternion_from_euler
        op_optimized = torch_optimizer(op)

        actual = op_optimized(roll, pitch, yaw)
        expected = op(roll, pitch, yaw)

        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1])
        self.assert_close(actual[2], expected[2])

    def test_forth_and_back(self, device, dtype):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        roll_new, pitch_new, yaw_new = euler_from_quaternion(qw, qx, qy, qz)
        self.assert_close(roll, roll_new)
        self.assert_close(pitch, pitch_new)
        self.assert_close(yaw, yaw_new)

    def test_values(self, device, dtype):
        # num_samples = 5
        # data = 2 * torch.rand(3, num_samples, device=device, dtype=dtype) - 1
        # roll, pitch, yaw = torch.pi * data
        roll = torch.tensor(
            [2.6518599987, 0.0612506270, 1.2417907715, 2.8829660416, -1.9961174726], device=device, dtype=dtype
        )

        pitch = torch.tensor(
            [2.3267219067, -2.7309591770, -1.4011553526, -2.1962766647, 2.1454355717], device=device, dtype=dtype
        )

        yaw = torch.tensor(
            [-0.8856627345, 0.2605336905, 0.4579202533, -1.3095731735, 0.6096843481], device=device, dtype=dtype
        )

        euler_expected = torch.tensor(
            [
                [-0.4897327125, 0.8148705959, 2.2559301853],
                [-3.0803420544, -0.4106334746, -2.8810589314],
                [1.2417914867, -1.4011553526, 0.4579201937],
                [-0.2586266696, -0.9453159571, 1.8320195675],
                [1.1454752684, 0.9961569905, -2.5319085121],
            ],
            device=device,
            dtype=dtype,
        )

        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        euler = euler_from_quaternion(qw, qx, qy, qz)
        euler = torch.stack(euler, -1)

        self.assert_close(euler, euler_expected, 1e-4, 1e-4)

        # this test is passing: pip install transforms3d
        # import transforms3d as tf3
        # out = [tf3.euler.euler2quat(roll[i], pitch[i], yaw[i]) for i in range(num_samples)]
        # out = torch.tensor(out, device=device, dtype=dtype)
        # self.assert_close(torch.stack((qw, qx, qy, qz), -1), out)

        # out = [tf3.euler.quat2euler((qw[i], qx[i], qy[i], qz[i])) for i in range(num_samples)]
        # out = torch.tensor(out, device=device, dtype=dtype)

    def test_convention_composition_is_rz_ry_rx(self, device, dtype):
        # Convention pin: "XYZ convention" in the docstring does not say whether the three
        # rotations are applied about the fixed axes or about the axes carried along by the body,
        # and the four candidate products differ enormously. The actual composition is
        #   R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        # i.e. extrinsic X -> Y -> Z about the FIXED axes (equivalently intrinsic Z-Y'-X''), with
        # roll about x, pitch about y and yaw about z. Three distinct non-symmetric angles are
        # required: a symmetric or single-axis input cannot separate the four candidates. Measured
        # max |R - candidate| at (0.3, 0.7, 1.1) in float64:
        #   Rz@Ry@Rx 0.0 (1.11e-16 when the product is built from math.cos/math.sin literals),
        #   Rx@Ry@Rz 0.6404683155788216, Ry@Rz@Rx 0.5484888138736672, Rx@Rz@Ry 0.2503184512807922.
        # The three rejected products are asserted to stay above 0.2 so the discrimination itself
        # is executable rather than a claim in a comment; at bfloat16, the coarsest dtype run, the
        # accepted product is still within 3.90625e-03 and the nearest rejected one at 0.25.
        # The return is a TUPLE of four separate tensors, not a stacked (*, 4) tensor; pinned first.
        # Snippet used to generate the elementary matrices (stdlib only):
        #   import math
        #   math.cos(0.3), math.sin(0.3) -> (0.955336489125606, 0.29552020666133955)
        #   math.cos(0.7), math.sin(0.7) -> (0.7648421872844885, 0.644217687237691)
        #   math.cos(1.1), math.sin(1.1) -> (0.4535961214255773, 0.8912073600614354)
        rot_x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.955336489125606, -0.29552020666133955],
                [0.0, 0.29552020666133955, 0.955336489125606],
            ],
            device=device,
            dtype=dtype,
        )
        rot_y = torch.tensor(
            [
                [0.7648421872844885, 0.0, 0.644217687237691],
                [0.0, 1.0, 0.0],
                [-0.644217687237691, 0.0, 0.7648421872844885],
            ],
            device=device,
            dtype=dtype,
        )
        rot_z = torch.tensor(
            [
                [0.4535961214255773, -0.8912073600614354, 0.0],
                [0.8912073600614354, 0.4535961214255773, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )

        quaternion = quaternion_from_euler(
            torch.tensor(0.3, device=device, dtype=dtype),
            torch.tensor(0.7, device=device, dtype=dtype),
            torch.tensor(1.1, device=device, dtype=dtype),
        )
        assert isinstance(quaternion, tuple)
        assert len(quaternion) == 4

        # .to(dtype) because quaternion_to_rotation_matrix returns float32 for float16/bfloat16
        # inputs; the cast keeps this pin about the composition order and nothing else.
        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.stack(quaternion)).to(dtype)

        self.assert_close(rot, rot_z @ rot_y @ rot_x)

        assert (rot - rot_x @ rot_y @ rot_z).abs().max() > 0.2
        assert (rot - rot_y @ rot_z @ rot_x).abs().max() > 0.2
        assert (rot - rot_x @ rot_z @ rot_y).abs().max() > 0.2


@pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
def test_vector_to_skew_symmetric_matrix(batch_size, device, dtype):
    if batch_size is None:
        vector = torch.rand(3, device=device, dtype=dtype)
    else:
        vector = torch.rand((batch_size, 3), device=device, dtype=dtype)
    skew_symmetric_matrix = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(vector)
    assert skew_symmetric_matrix.shape[-1] == 3
    assert skew_symmetric_matrix.shape[-2] == 3
    z = torch.zeros_like(vector[..., 0])
    assert_close(skew_symmetric_matrix[..., 0, 0], z)
    assert_close(skew_symmetric_matrix[..., 1, 1], z)
    assert_close(skew_symmetric_matrix[..., 2, 2], z)
    assert_close(skew_symmetric_matrix[..., 0, 1], -vector[..., 2])
    assert_close(skew_symmetric_matrix[..., 1, 0], vector[..., 2])
    assert_close(skew_symmetric_matrix[..., 0, 2], vector[..., 1])
    assert_close(skew_symmetric_matrix[..., 2, 0], -vector[..., 1])
    assert_close(skew_symmetric_matrix[..., 1, 2], -vector[..., 0])
    assert_close(skew_symmetric_matrix[..., 2, 1], vector[..., 0])

    # Convention's enforcement point: [v]x @ x == cross(v, x) -- the vector is the LEFT factor
    # of the cross product, NOT cross(x, v), which is the negation.
    # Snippet used to generate expected (stdlib only):
    #   v, x = (1, 2, 3), (4, 5, 6)
    #   cross(v, x) = (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) -> (-3, 6, -3)
    v = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
    x = torch.tensor([4.0, 5.0, 6.0], device=device, dtype=dtype)
    skew = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(v)
    expected_cross = torch.tensor([-3.0, 6.0, -3.0], device=device, dtype=dtype)
    assert_close(skew @ x, expected_cross)


class TestAxisAngleToRotationMatrix:
    def test_identity_rotation(self):
        aa = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
        R = axis_angle_to_rotation_matrix(aa)
        Id = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        assert torch.allclose(R, Id, atol=1e-6)

    def test_90deg_x_axis(self):
        aa = torch.tensor([[torch.pi / 2, 0.0, 0.0]], dtype=torch.float64)
        R = axis_angle_to_rotation_matrix(aa).squeeze(0)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(R, expected, atol=1e-6)

    def test_180deg_y_axis(self):
        aa = torch.tensor([[0.0, torch.pi, 0.0]], dtype=torch.float64)
        R = axis_angle_to_rotation_matrix(aa).squeeze(0)
        expected = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(R, expected, atol=1e-6)

    def test_batched_input(self):
        aa = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [torch.pi / 2, 0.0, 0.0],
                [0.0, torch.pi, 0.0],
            ],
            dtype=torch.float64,
        )
        R = axis_angle_to_rotation_matrix(aa)
        assert R.shape == (3, 3, 3)
