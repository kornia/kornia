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

import math
import subprocess
import sys
import textwrap

import pytest
import torch

import kornia
import kornia.geometry.transform.imgwarp

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available
from testing.geometry.create import create_random_homography


class TestAngleToRotationMatrix(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4).to(device)
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(inp)
        assert rotmat.shape == (1, 3, 4, 4, 2, 2)

    def test_angles(self, device):
        ang_deg = torch.tensor([0, 90.0], device=device)
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0, 1.0], [-1.0, 0]]], device=device)
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(ang_deg)
        self.assert_close(rotmat, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.imgwarp.angle_to_rotation_matrix, (img,))

    @pytest.mark.skip("Problems with kornia.pi")
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix
        model_jit = torch.jit.script(kornia.geometry.transform.imgwarp.angle_to_rotation_matrix)
        self.assert_close(model(patches), model_jit(patches))


class TestGetLAFScale(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        rotmat = kornia.feature.get_laf_scale(inp)
        assert rotmat.shape == (1, 3, 1, 1)

    def test_scale(self, device):
        inp = torch.tensor([[5.0, 1, 0], [1, 1, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[2]]]], device=device).float()
        rotmat = kornia.feature.get_laf_scale(inp)
        self.assert_close(rotmat, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.get_laf_scale, (img,))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.get_laf_scale
        model_jit = torch.jit.script(kornia.feature.get_laf_scale)
        self.assert_close(model(img), model_jit(img))


class TestGetLAFCenter(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        xy = kornia.feature.get_laf_center(inp)
        assert xy.shape == (1, 3, 2)

    def test_center(self, device):
        inp = torch.tensor([[5.0, 1, 2], [1, 1, 3]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[2, 3]]], device=device).float()
        xy = kornia.feature.get_laf_center(inp)
        self.assert_close(xy, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.get_laf_center, (img,))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.get_laf_center
        model_jit = torch.jit.script(kornia.feature.get_laf_center)
        self.assert_close(model(img), model_jit(img))


class TestGetLAFOri(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        ori = kornia.feature.get_laf_orientation(inp)
        assert ori.shape == (1, 3, 1)

    def test_ori(self, device):
        inp = torch.tensor([[1, 1, 2], [1, 1, 3]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[45.0]]], device=device).float()
        angle = kornia.feature.get_laf_orientation(inp)
        self.assert_close(angle, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.get_laf_orientation, (img,))

    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.get_laf_orientation
        model_jit = torch.jit.script(kornia.feature.get_laf_orientation)
        self.assert_close(model(img), model_jit(img))


class TestScaleLAF(BaseTester):
    def test_shape_float(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        scale = 23.0
        assert kornia.feature.scale_laf(inp, scale).shape == inp.shape

    def test_shape_tensor(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        scale = torch.zeros(7, 1, 1, 1, device=device).float()
        assert kornia.feature.scale_laf(inp, scale).shape == inp.shape

    def test_scale(self, device):
        inp = torch.tensor([[5.0, 1, 0.8], [1, 1, -4.0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        scale = torch.tensor([[[[2.0]]]], device=device).float()
        out = kornia.feature.scale_laf(inp, scale)
        expected = torch.tensor([[[[10.0, 2, 0.8], [2, 2, -4.0]]]], device=device).float()
        self.assert_close(out, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        scale = torch.rand(batch_size, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.scale_laf, (laf, scale), atol=1e-4)

    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        scale = torch.rand(batch_size, device=device)
        model = kornia.feature.scale_laf
        model_jit = torch.jit.script(kornia.feature.scale_laf)
        self.assert_close(model(laf, scale), model_jit(laf, scale))


class TestSetLAFOri(BaseTester):
    def test_shape_tensor(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        ori = torch.ones(7, 3, 1, 1, device=device).float()
        assert kornia.feature.set_laf_orientation(inp, ori).shape == inp.shape

    def test_ori(self, device):
        inp = torch.tensor([[0.0, 5.0, 0.8], [-5.0, 0, -4.0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        ori = torch.zeros(1, 1, 1, 1, device=device).float()
        out = kornia.feature.set_laf_orientation(inp, ori)
        expected = torch.tensor([[[[5.0, 0.0, 0.8], [0.0, 5.0, -4.0]]]], device=device).float()
        self.assert_close(out, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        ori = torch.rand(batch_size, channels, 1, 1, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.set_laf_orientation, (laf, ori), atol=1e-4)

    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        ori = torch.rand(batch_size, channels, 1, 1, device=device)
        model = kornia.feature.set_laf_orientation
        model_jit = torch.jit.script(kornia.feature.set_laf_orientation)
        self.assert_close(model(laf, ori), model_jit(laf, ori))


class TestMakeUpright(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        rotmat = kornia.feature.make_upright(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_do_nothing(self, device):
        inp = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[1, 0, 0], [0, 1, 0]]]], device=device).float()
        laf = kornia.feature.make_upright(inp)
        self.assert_close(laf, expected)

    def test_do_nothing_with_scalea(self, device):
        inp = torch.tensor([[2, 0, 0], [0, 2, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[2, 0, 0], [0, 2, 0]]]], device=device).float()
        laf = kornia.feature.make_upright(inp)
        self.assert_close(laf, expected)

    def test_check_zeros(self, device):
        inp = torch.rand(4, 5, 2, 3, device=device)
        laf = kornia.feature.make_upright(inp)
        must_be_zeros = laf[:, :, 0, 1]
        self.assert_close(must_be_zeros, torch.zeros_like(must_be_zeros))

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 14, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.make_upright, (img,))

    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.make_upright
        model_jit = torch.jit.script(kornia.feature.make_upright)
        self.assert_close(model(img), model_jit(img))


class TestELL2LAF(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(5, 3, 5, device=device)
        inp[:, :, 3] = 0
        rotmat = kornia.feature.ellipse_to_laf(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_conversion(self, device):
        inp = torch.tensor([[10, -20, 0.01, 0, 0.01]], device=device).float()
        inp = inp.view(1, 1, 5)
        expected = torch.tensor([[10, 0, 10.0], [0, 10, -20]], device=device).float()
        expected = expected.view(1, 1, 2, 3)
        laf = kornia.feature.ellipse_to_laf(inp)
        self.assert_close(laf, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height = 1, 2, 5
        img = torch.rand(batch_size, channels, height, device=device, dtype=torch.float64).abs()
        img[:, :, 2] = img[:, :, 3].abs() + 0.3
        img[:, :, 4] += 1.0
        # assure it is positive definite
        self.gradcheck(kornia.feature.ellipse_to_laf, (img,))

    def test_small_root_sum_is_not_clamped(self, device):
        # The root sum is finite and nonzero, so clamping it changes a valid inverse by orders of magnitude.
        tiny = torch.finfo(torch.float32).tiny
        inp = torch.tensor([[[0.0, 0.0, tiny, tiny, tiny]]], device=device, dtype=torch.float32)
        expected = torch.tensor(-0.5 / math.sqrt(tiny), device=device, dtype=torch.float32)
        laf = kornia.feature.ellipse_to_laf(inp)
        assert torch.isfinite(laf).all()
        self.assert_close(laf[0, 0, 1, 0], expected)

    def test_no_overflow_asymmetric_diag(self, device, dtype):
        # Regression test: the closed-form inverse's off-diagonal must divide by the root product,
        # not multiply reciprocals. `a` is the dtype's smallest normal and `c` is picked so that
        # (sqrt(a) + sqrt(c)) * sqrt(a) == 0.5, which makes the intermediate of the fixed order
        # `-a21 * (1 / a11) * (1 / a22)` twice b -- inf for a b above half the dtype's maximum,
        # although the result itself is well inside range. https://github.com/kornia/kornia/pull/4122
        finfo = torch.finfo(dtype)
        a11 = math.sqrt(finfo.tiny)
        a22 = 0.5 / a11
        inp = torch.tensor([[[0.0, 0.0, finfo.tiny, finfo.max * 0.75, a22 * a22]]], device=device, dtype=dtype)
        # Reference in float64, from the inputs as the dtype actually rounded them. Via CPU:
        # MPS tensors cannot be converted to float64 (TESTING.md, "Writing new tests that work on MPS").
        expected = kornia.feature.ellipse_to_laf(inp.cpu().double())[0, 0, 1, 0]
        laf = kornia.feature.ellipse_to_laf(inp)
        assert torch.isfinite(laf).all()
        self.assert_close(laf[0, 0, 1, 0], expected.to(device=device, dtype=dtype))

    def test_no_overflow_subnormal_diag(self, device, dtype):
        # The mirror case: a subnormal but nondegenerate diagonal makes 1 / (a11 * a22) overflow, so
        # forming that product first turns a mathematically-zero off-diagonal into 0 * inf = nan.
        # Subnormal input is unavoidable here -- a11 * a22 >= finfo.tiny > 1 / finfo.max in every IEEE
        # dtype, so no normal input reaches this regime -- hence the flush-to-zero guard below.
        finfo = torch.finfo(dtype)
        a = finfo.tiny / 8  # subnormal, and small enough that 1 / (a11 * a22) == 1 / a overflows
        inp = torch.tensor([[[0.0, 0.0, a, 0.0, a]]], device=device, dtype=dtype)
        # Guard on the arithmetic, not just the storage: a backend can store the subnormal
        # faithfully yet flush it inside sqrt, which makes the diagonal degenerate all the same.
        if inp[0, 0, 2].sqrt() == 0:
            pytest.skip("backend flushes subnormals to zero, so this diagonal is degenerate here")
        laf = kornia.feature.ellipse_to_laf(inp)
        assert torch.isfinite(laf).all()
        self.assert_close(laf[0, 0, 1, 0], torch.zeros_like(laf[0, 0, 1, 0]))

    def test_no_underflow_asymmetric_diag(self, device, dtype):
        # Regression test for the ordering the two tests above do not cover: multiplying by the
        # smaller reciprocal first (f7b573a3, since replaced by the division form) passes both of
        # them but silently flushes a representable off-diagonal to a false zero. `a` is the
        # dtype's smallest normal and `c` its reciprocal squared, so a11 * a22 == 1 and
        # inv22 == a11 is itself tiny; `b` is picked so a21 * inv22 -- the min/max order's first
        # product -- underflows to zero while the true off-diagonal, a21 / (a11 * a22), stays
        # representable. https://github.com/kornia/kornia/pull/4122
        finfo = torch.finfo(dtype)
        a11 = math.sqrt(finfo.tiny)
        a22 = 1.0 / a11
        b = finfo.eps * math.sqrt(a11)
        inp = torch.tensor([[[0.0, 0.0, finfo.tiny, b, a22 * a22]]], device=device, dtype=dtype)
        # Guard on the arithmetic, not just the storage: a21 is shared by every ordering, so if a
        # backend's division already flushes it to zero, no ordering has anything left to get wrong.
        a11_t, a22_t = inp[..., 2:3].abs().sqrt(), inp[..., 4:5].abs().sqrt()
        a21_t = inp[..., 3:4] / (a11_t + a22_t)
        if (a21_t == 0).any():
            pytest.skip("backend flushes this off-diagonal's shared numerator to zero regardless of ordering")
        expected = kornia.feature.ellipse_to_laf(inp.cpu().double())[0, 0, 1, 0]
        laf = kornia.feature.ellipse_to_laf(inp)
        assert torch.isfinite(laf).all()
        # Loose relative tolerance: the point is distinguishing a false zero (100% off) from the
        # real value, not pinning float16's reduced precision this deep into its subnormal range.
        self.assert_close(laf[0, 0, 1, 0], expected.to(device=device, dtype=dtype), rtol=0.1, atol=0.0)

    @pytest.mark.parametrize("degenerate_index", [2, 4])
    def test_degenerate_ellipse_is_non_finite(self, device, dtype, degenerate_index):
        # A degenerate ellipse makes the matrix singular. The batched torch.inverse this replaced
        # raised linalg.LinAlgError; the closed form returns non-finite values and must not raise.
        inp = torch.tensor([[[1.0, 2.0, 3.0, 0.5, 4.0]]], device=device, dtype=dtype)
        inp[0, 0, degenerate_index] = 0.0
        laf = kornia.feature.ellipse_to_laf(inp)
        assert not torch.isfinite(laf[0, 0, :, :2]).all()
        self.assert_close(laf[0, 0, :, 2], inp[0, 0, :2])  # the centre is untouched

    def test_dynamo(self, device, dtype, torch_optimizer):
        inp = self._well_conditioned_ellipses(device, dtype)
        op = kornia.feature.ellipse_to_laf
        self.assert_close(torch_optimizer(op)(inp), op(inp))

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_dynamo_fullgraph(self, device, dtype):
        # The batched torch.inverse this replaced graph-broke; the closed form must stay capturable.
        inp = self._well_conditioned_ellipses(device, dtype)
        expected = kornia.feature.ellipse_to_laf(inp)
        torch._dynamo.reset()
        compiled = torch.compile(kornia.feature.ellipse_to_laf, fullgraph=True)
        self.assert_close(compiled(inp), expected)

    @staticmethod
    def _well_conditioned_ellipses(device, dtype):
        inp = torch.rand(1, 2, 5, device=device, dtype=dtype).abs()
        inp[..., 2] = inp[..., 3].abs() + 0.3
        inp[..., 4] += 1.0
        return inp

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        batch_size, channels, height = 1, 2, 5
        img = torch.rand(batch_size, channels, height, device=device).abs()
        img[:, :, 2] = img[:, :, 3].abs() + 0.3
        img[:, :, 4] += 1.0
        model = kornia.feature.ellipse_to_laf
        model_jit = torch.jit.script(kornia.feature.ellipse_to_laf)
        self.assert_close(model(img), model_jit(img))


class TestLAFIsValid(BaseTester):
    def test_finite_nonsingular_laf_is_valid(self, device, dtype):
        laf = torch.tensor([[[[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]]], device=device, dtype=dtype)
        assert kornia.feature.laf_is_valid(laf).all()

    def test_nonfinite_or_singular_laf_is_invalid(self, device, dtype):
        laf = torch.tensor(
            [[[[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]], [[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )
        laf[0, 0, 0, 0] = torch.inf
        expected = torch.tensor([[False, False]], device=device)
        assert torch.equal(kornia.feature.laf_is_valid(laf), expected)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_dynamo_fullgraph(self, device, dtype):
        laf = torch.tensor([[[[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]]], device=device, dtype=dtype)
        expected = kornia.feature.laf_is_valid(laf)
        torch._dynamo.reset()
        compiled = torch.compile(kornia.feature.laf_is_valid, fullgraph=True)
        assert torch.equal(compiled(laf), expected)


class TestNormalizeLAF(BaseTester):
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3)
        img = torch.rand(5, 3, 10, 10)
        assert inp.shape == kornia.feature.normalize_laf(inp, img).shape

    def test_roundtrip_non_square_wide(self, device, dtype):
        # Wide image (W >> H): x/y coords are normalized differently from scale components.
        # The normalize→denormalize round-trip must be exact.
        laf = torch.tensor([[10.0, 0.0, 160.0], [0.0, 10.0, 60.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        img = torch.zeros(1, 1, 120, 320, device=device, dtype=dtype)
        laf_norm = kornia.feature.normalize_laf(laf, img)
        laf_back = kornia.feature.denormalize_laf(laf_norm, img)
        self.assert_close(laf_back, laf)

    def test_roundtrip_non_square_tall(self, device, dtype):
        # Tall image (H >> W): verify round-trip in the opposite aspect ratio.
        laf = torch.tensor([[5.0, 0.0, 40.0], [0.0, 5.0, 100.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        img = torch.zeros(1, 1, 240, 80, device=device, dtype=dtype)
        laf_norm = kornia.feature.normalize_laf(laf, img)
        laf_back = kornia.feature.denormalize_laf(laf_norm, img)
        self.assert_close(laf_back, laf)

    def test_conversion(self, device):
        w, h = 9, 5
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]]).float()
        laf = laf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w)
        expected = torch.tensor([[[[0.25, 0, 0.125], [0, 0.25, 0.25]]]]).float()
        lafn = kornia.feature.normalize_laf(laf, img)
        self.assert_close(lafn, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        img = torch.rand(batch_size, 3, 10, 32, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.normalize_laf, (laf, img))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        model = kornia.feature.normalize_laf
        model_jit = torch.jit.script(kornia.feature.normalize_laf)
        self.assert_close(model(laf, img), model_jit(laf, img))


class TestLAF2pts(BaseTester):
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        n_pts = 13
        assert kornia.feature.laf_to_boundary_points(inp, n_pts).shape == (5, 3, n_pts, 2)

    def test_conversion(self, device):
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]], device=device).float()
        laf = laf.view(1, 1, 2, 3)
        n_pts = 6
        expected = torch.tensor([[[[1, 1], [1, 2], [2, 1], [1, 0], [0, 1], [1, 2]]]], device=device).float()
        pts = kornia.feature.laf_to_boundary_points(laf, n_pts)
        self.assert_close(pts, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.laf_to_boundary_points, (laf))

    def test_matches_explicit_affine_reference(self, device, dtype):
        """The boundary points are the LAF affine map applied to the origin + unit-circle basis.

        `test_conversion` uses a symmetric 2x2 block, so it passes even if the LAF and the basis are
        multiplied in the wrong order. This uses random anisotropic LAFs, where they disagree, and
        pins the equality that lets the implementation multiply by the (2, 3) LAF directly instead
        of appending a constant `[0, 0, 1]` row and dividing the result by its homogeneous
        coordinate, which is then always exactly 1.
        """
        if dtype in (torch.float16, torch.bfloat16):
            # `einsum` and `bmm` accumulate the length-3 dot in a different order, which in half
            # precision costs more than the equality being pinned here (4.4e-3 in float16 against a
            # float64 reference, i.e. this dtype's own resolution). `test_dtype_device_preserved`
            # carries the half coverage.
            pytest.skip("reference accumulates differently from bmm in half precision")
        torch.manual_seed(0)
        laf = torch.randn(2, 5, 2, 3, device=device, dtype=dtype)
        n_pts = 9
        # The implementation builds its angles in float32 and casts, so the reference does too.
        angles = torch.linspace(0, 2 * math.pi, n_pts - 1, device=device).to(dtype)
        origin = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
        basis = torch.cat([origin, torch.stack([angles.sin(), angles.cos(), torch.ones_like(angles)], dim=-1)])
        expected = torch.einsum("bnij,pj->bnpi", laf, basis)
        self.assert_close(kornia.feature.laf_to_boundary_points(laf, n_pts), expected)

    def test_dtype_device_preserved(self, device, dtype):
        """Boundary points keep the LAF's dtype and device, in every dtype.

        Nothing else covered this op in half precision. Note this cannot see *where* the basis is
        cast -- casting it before or after the broadcast gives the same dtype, device and values,
        so that ordering is a memory property, not an observable one.
        """
        laf = torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        n_pts = 6
        expected = torch.tensor([[[[1, 1], [1, 2], [2, 1], [1, 0], [0, 1], [1, 2]]]], device=device, dtype=dtype)
        pts = kornia.feature.laf_to_boundary_points(laf, n_pts)
        assert pts.dtype == dtype
        assert pts.device == laf.device
        self.assert_close(pts, expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        laf = torch.rand(2, 4, 2, 3, device=device, dtype=dtype)
        expected = kornia.feature.laf_to_boundary_points(laf)
        op = torch_optimizer(kornia.feature.laf_to_boundary_points)
        self.assert_close(op(laf), expected)

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 3, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.laf_to_boundary_points
        model_jit = torch.jit.script(kornia.feature.laf_to_boundary_points)
        self.assert_close(model(laf), model_jit(laf))


class TestDenormalizeLAF(BaseTester):
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        assert inp.shape == kornia.feature.denormalize_laf(inp, img).shape

    def test_conversion(self, device):
        w, h = 9, 5
        expected = torch.tensor([[1, 0, 1], [0, 1, 1]], device=device).float()
        expected = expected.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w, device=device)
        lafn = torch.tensor([[0.25, 0, 0.125], [0, 0.25, 0.25]], device=device).float()
        laf = kornia.feature.denormalize_laf(lafn.view(1, 1, 2, 3), img)
        self.assert_close(laf, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        img = torch.rand(batch_size, 3, 10, 32, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.denormalize_laf, (laf, img))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        model = kornia.feature.denormalize_laf
        model_jit = torch.jit.script(kornia.feature.denormalize_laf)
        self.assert_close(model(laf, img), model_jit(laf, img))


class TestGenPatchGrid(BaseTester):
    def test_shape(self, device):
        laf = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF

        grid = generate_patch_grid_from_normalized_LAF(img, laf, PS)
        assert grid.shape == (15, 3, 3, 2)

    def test_gradcheck(self, device):
        laf = torch.rand(5, 3, 2, 3, device=device, dtype=torch.float64)
        img = torch.rand(5, 3, 10, 10, device=device, dtype=torch.float64)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF

        self.gradcheck(generate_patch_grid_from_normalized_LAF, (img, laf, PS))


class TestClampGridToPixelCenters(BaseTester):
    """Pin the MPS ``padding_mode="border"`` emulation used by the patch extractors.

    MPS has no ``border`` padding for ``grid_sample``, so the extractors clamp the grid and use
    zero padding instead. The clamp must target the outermost pixel *center*; clamping to
    :math:`\\pm 1` lands on the outer *edge* of the border pixel and halves its value.

    The reference is always computed on CPU, because ``padding_mode="border"`` is exactly what the
    target device may not support.
    """

    @staticmethod
    def _make(h, w, dtype):
        torch.manual_seed(0)
        img = torch.rand(1, 2, h, w, dtype=dtype)
        # a grid that deliberately runs off every edge
        grid = (torch.rand(1, 9, 11, 2, dtype=dtype) * 3.0) - 1.5
        return img, grid

    def test_matches_border_padding(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("grid_sample reference comparison is meaningful only in full precision")
        h, w = 17, 23
        img, grid = self._make(h, w, dtype)
        expected = torch.nn.functional.grid_sample(img, grid, padding_mode="border", align_corners=False)

        clamped = kornia.feature.laf._clamp_grid_to_pixel_centers(grid.to(device), h, w)
        actual = torch.nn.functional.grid_sample(img.to(device), clamped, padding_mode="zeros", align_corners=False)
        self.assert_close(actual.cpu(), expected)

    def test_naive_clamp_is_not_equivalent(self, device, dtype):
        """Guard against regressing to ``grid.clamp(-1, 1)``, which is off by half a pixel."""
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("grid_sample reference comparison is meaningful only in full precision")
        h, w = 17, 23
        img, grid = self._make(h, w, dtype)
        expected = torch.nn.functional.grid_sample(img, grid, padding_mode="border", align_corners=False)
        naive = torch.nn.functional.grid_sample(
            img.to(device), grid.clamp(-1, 1).to(device), padding_mode="zeros", align_corners=False
        )
        assert (naive.cpu() - expected).abs().max() > 1e-3

    def test_shape(self, device, dtype):
        grid = torch.rand(4, 5, 6, 2, device=device, dtype=dtype)
        assert kornia.feature.laf._clamp_grid_to_pixel_centers(grid, 8, 9).shape == grid.shape

    def test_gradcheck(self, device):
        grid = ((torch.rand(1, 3, 3, 2, device=device, dtype=torch.float64) * 3.0) - 1.5).requires_grad_()
        self.gradcheck(lambda g: kornia.feature.laf._clamp_grid_to_pixel_centers(g, 7, 11), (grid,), fast_mode=False)

    @pytest.mark.parametrize(
        "extractor",
        [kornia.feature.extract_patches_simple, kornia.feature.extract_patches_from_pyramid],
    )
    def test_extractors_match_cpu_at_the_border(self, device, dtype, extractor):
        """Patches overlapping the image border must not depend on the device."""
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("cross-device comparison is meaningful only in full precision")
        torch.manual_seed(0)
        img = torch.rand(1, 1, 64, 64, dtype=dtype)
        # centered near the corner with a large scale, so the patch runs off two edges
        laf = kornia.feature.laf_from_center_scale_ori(
            torch.tensor([[[3.0, 3.0]]], dtype=dtype), torch.full((1, 1, 1, 1), 12.0, dtype=dtype)
        )
        expected = extractor(img, laf, 32)
        actual = extractor(img.to(device), laf.to(device), 32)
        self.assert_close(actual.cpu(), expected)


# The two half-precision border tests below are CPU-only, for two reasons. The broken kernel and
# the float32 workaround that routes around it are CPU-specific, so no other device exercises
# anything the rest of the suite misses; and the half dtype comes from a `half_dtype` parameter,
# which the CUDA half-precision isolation in `conftest.py` does not see -- both
# `_is_subprocess_isolated_test` and `skip_half_precision_on_cuda` key on the global `dtype`
# fixture (`dtype_name`). A `--device=cuda --dtype=float32` job would therefore run half kernels
# in the shared CUDA context with no subprocess isolation, where a device-side assert poisons
# every later test in the process. MPS autocast changes the effective dtype as well.
_HALF_BORDER_SKIP = "the reduced-precision `grid_sample` gap and its float32 workaround are CPU-only"


def _corner_border_laf(device, dtype, scale: float = 8.0, angle_deg: float = 15.0, center: float = 255.0):
    """Build a rotated LAF centered on the last pixel of a 256x256 image.

    Its patch straddles the image corner, so half of the sampling grid falls outside the frame and
    is served by the border padding. ``scale`` is below the patch size used by the tests (16), so
    ``extract_patches_from_pyramid`` routes it through pyramid level 0: at a downsampled level the
    repeated blur turns the patch into a near-constant that hides a bad read.
    """
    t = math.radians(angle_deg)
    cos, sin = scale * math.cos(t), scale * math.sin(t)
    laf = torch.tensor([[cos, -sin, center], [sin, cos, center]], device=device, dtype=dtype)
    return laf.view(1, 1, 2, 3).expand(1, 3, 2, 3)


class TestExtractPatchesSimple(BaseTester):
    def test_shape(self, device):
        laf = torch.rand(5, 4, 2, 3, device=device)
        img = torch.rand(5, 3, 100, 30, device=device)
        PS = 10
        patches = kornia.feature.extract_patches_simple(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    def test_non_zero(self, device):
        img = torch.zeros(1, 1, 24, 24, device=device)
        img[:, :, 10:, 20:] = 1.0
        laf = torch.tensor([[8.0, 0, 14.0], [0, 8.0, 8.0]], device=device).reshape(1, 1, 2, 3)

        PS = 32
        patches = kornia.feature.extract_patches_simple(img, laf, PS)
        assert patches.mean().item() > 0.01
        assert patches.shape == (1, 1, 1, PS, PS)

    def test_exception(self, device, dtype):
        # Batch disagreement is rejected at the boundary with a clear message instead of an
        # internal `grid_sample` error or a silently truncated result.
        img = torch.rand(3, 1, 32, 32, device=device, dtype=dtype)
        with pytest.raises(Exception, match="same batch size"):
            kornia.feature.extract_patches_simple(img, torch.rand(1, 2, 2, 3, device=device, dtype=dtype), 8)

    def test_mixed_dtype_laf_under_autocast(self, device):
        # Reduced-precision detectors commonly emit a half/bfloat16 LAF while their source image
        # remains float32. The extractor owns this small conversion instead of rejecting a valid
        # autocast pipeline or relying on backend-specific grid_sample promotion.
        if device.type not in ("cpu", "cuda"):
            pytest.skip("autocast coverage is provided by CPU and CUDA")
        laf_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        img = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.tensor([[4.0, 0.0, 16.0], [0.0, 4.0, 16.0]], device=device, dtype=laf_dtype).view(1, 1, 2, 3)
        with torch.autocast(device_type=device.type, dtype=laf_dtype):
            actual = kornia.feature.extract_patches_simple(img, laf, 8)
            expected = kornia.feature.extract_patches_simple(img, laf.float(), 8)
        assert actual.dtype == img.dtype
        self.assert_close(actual, expected)

    def test_mixed_dtype_preserves_laf_precision(self, device):
        # A float32 LAF paired with a half image must not be rounded to half and then upcast: its
        # subpixel coordinates are the reason grid arithmetic uses a promoted dtype.
        if device.type != "cpu":
            pytest.skip("CPU provides deterministic reduced-precision reference coverage")
        img = torch.rand(1, 1, 32, 32).to(torch.float16)
        laf = torch.tensor([[4.1234, 0.2712, 15.789], [-0.1923, 3.9876, 16.321]]).view(1, 1, 2, 3)
        actual = kornia.feature.extract_patches_simple(img, laf, 8)
        expected = kornia.feature.extract_patches_simple(img.float(), laf, 8).to(img.dtype)
        assert actual.dtype == img.dtype
        assert torch.equal(actual, expected)

    def test_nonfinite_laf_returns_zero_patch_with_safe_backward(self, device, dtype):
        # Same contract as `TestExtractPatchesPyr`: a training-time detector can emit a non-finite
        # LAF frame, including one whose only invalid entry is its center. The whole frame is
        # sanitized before any grid arithmetic, so it returns a zero patch and a zero gradient
        # rather than handing `grid_sample` an invalid grid -- whose CPU border-padding backward
        # kernel can segfault the process. Infinity is invalid for the same API-level reason.
        PS = 8
        img = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        nan = float("nan")
        inf = float("inf")
        laf = torch.tensor(
            [
                [[8.0, 0.0, 32.0], [0.0, 8.0, 32.0]],
                [[nan, nan, nan], [nan, nan, nan]],
                [[8.0, 0.0, nan], [0.0, 8.0, nan]],
                [[8.0, 0.0, inf], [0.0, 8.0, inf]],
            ],
            device=device,
            dtype=dtype,
        ).view(1, 4, 2, 3)

        expected_finite = kornia.feature.extract_patches_simple(img, laf[:, :1], PS)
        grad_img = img.detach().clone().requires_grad_()
        grad_laf = laf.detach().clone().requires_grad_()
        patches = kornia.feature.extract_patches_simple(grad_img, grad_laf, PS)
        self.assert_close(patches[:, :1], expected_finite)
        assert patches[:, 1:].abs().sum().item() == 0
        patches.sum().backward()
        assert grad_img.grad is not None and bool(grad_img.grad.isfinite().all())
        assert grad_laf.grad is not None and bool(grad_laf.grad.isfinite().all())
        assert grad_laf.grad[:, 1:].abs().sum().item() == 0

    def test_same_odd(self, device, dtype):
        img = torch.arange(5)[None].repeat(5, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[2.0, 0, 2.0], [0, 2.0, 2.0]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_simple(img, laf, 5, 1.0)
        self.assert_close(img, patch[0])

    def test_same_even(self, device, dtype):
        img = torch.arange(4)[None].repeat(4, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[1.5, 0, 1.5], [0, 1.5, 1.5]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_simple(img, laf, 4, 1.0)
        self.assert_close(img, patch[0])

    def test_gradcheck(self, device):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]], device=device, dtype=torch.float64)
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 20, 30, device=device, dtype=torch.float64)
        PS = 11
        self.gradcheck(kornia.feature.extract_patches_simple, (img, nlaf, PS, False), fast_mode=False)

    def test_batch_independence(self, device, dtype):
        # Each patch must come only from its own batch element.
        B, N, PS = 3, 4, 8
        img = torch.arange(B, device=device, dtype=dtype).view(B, 1, 1, 1).expand(B, 1, 32, 32).contiguous()
        laf = torch.tensor([[4.0, 0.0, 16.0], [0.0, 4.0, 16.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        patches = kornia.feature.extract_patches_simple(img, laf.expand(B, N, 2, 3), PS)
        assert patches.shape == (B, N, 1, PS, PS)
        expected = torch.arange(B, device=device, dtype=dtype).view(B, 1, 1, 1, 1).expand(B, N, 1, PS, PS)
        self.assert_close(patches, expected)

    def test_empty(self, device, dtype):
        # Degenerate shapes follow the empty-in -> empty-out convention instead of raising from
        # inside `affine_grid`.
        PS = 8
        patches = kornia.feature.extract_patches_simple(
            torch.rand(0, 2, 32, 32, device=device, dtype=dtype), torch.rand(0, 3, 2, 3, device=device, dtype=dtype), PS
        )
        assert patches.shape == (0, 3, 2, PS, PS)
        assert patches.dtype == dtype
        patches = kornia.feature.extract_patches_simple(
            torch.rand(2, 1, 32, 32, device=device, dtype=dtype), torch.rand(2, 0, 2, 3, device=device, dtype=dtype), PS
        )
        assert patches.shape == (2, 0, 1, PS, PS)

    def test_chunked_matches_single_call(self, device, dtype, monkeypatch):
        # Forcing one LAF per chunk must reproduce the single-call result exactly, since
        # chunking only splits the grid along N.
        import kornia.feature.laf as laf_module

        torch.manual_seed(0)
        img = torch.rand(2, 3, 48, 64, device=device, dtype=dtype)
        laf = torch.rand(2, 5, 2, 3, device=device, dtype=dtype)
        expected = kornia.feature.extract_patches_simple(img, laf, 8)
        monkeypatch.setattr(laf_module, "_grid_chunk_lafs", lambda *args: 1)
        chunked = kornia.feature.extract_patches_simple(img, laf, 8)
        self.assert_close(chunked, expected)
        if device.type == "cpu":
            assert torch.equal(chunked, expected)

    def test_chunk_budget_accounts_for_channels(self):
        # A high-channel feature map can make the sampled chunk much larger than its 2-coordinate
        # grid. Both transient tensors must respect the workspace budget.
        import kornia.feature.laf as laf_module

        assert laf_module._grid_chunk_lafs(1, 1000, 1, 32, 4) == 1000
        assert laf_module._grid_chunk_lafs(1, 1000, 256, 32, 4) == 64

    def test_channel_laf_correspondence(self, device, dtype):
        # The patch at [b, n] must equal the one extracted for that LAF alone. At ch=1 or N=1 a
        # scrambled unfold is bit-identical to the right one (it permutes a size-1 dim), so this
        # pins ch>1 together with N>1 and distinct LAFs.
        B, N, ch, PS = 2, 4, 3, 8
        torch.manual_seed(0)
        img = torch.rand(B, ch, 48, 48, device=device, dtype=dtype)
        laf = torch.rand(B, N, 2, 3, device=device, dtype=dtype)
        patches = kornia.feature.extract_patches_simple(img, laf, PS)
        for b in range(B):
            for n in range(N):
                single = kornia.feature.extract_patches_simple(img[b : b + 1], laf[b : b + 1, n : n + 1], PS)
                self.assert_close(patches[b, n], single[0, 0])

    @pytest.mark.parametrize("half_dtype", [torch.float16, torch.bfloat16])
    def test_border_patches_stay_in_range(self, device, half_dtype):
        # A patch straddling the frame reads border pixels, so every sampled value is a convex
        # combination of image values and cannot leave the image's range. The float16/bfloat16 CPU
        # `grid_sample` kernels in torch <= 2.9 read out of bounds for such coordinates and return
        # zeros, values of order 1e4 or NaN depending on what the heap happens to hold, so the
        # extractor samples reduced-precision CPU images in float32. Parametrized on the dtype
        # rather than taking the global `dtype` fixture: CI runs the suite in float32/float64 only,
        # where the kernel is sound and this test cannot fail.
        if device.type != "cpu":
            pytest.skip(_HALF_BORDER_SKIP)
        torch.manual_seed(0)
        img = torch.rand(1, 1, 256, 256, device=device).to(half_dtype)
        laf = _corner_border_laf(device, half_dtype)
        patches = kornia.feature.extract_patches_simple(img, laf, 16)
        assert patches.dtype == half_dtype
        assert bool(patches.isfinite().all())
        assert patches.min() >= img.min()
        assert patches.max() <= img.max()

    def test_laf_on_another_device(self, device, dtype):
        # The sampling grid follows the LAF, so it has to be moved to the image: this function has
        # always accepted a LAF on a different device than the image and returned a patch on the
        # image's device.
        if device.type == "cpu":
            pytest.skip("needs a second device besides the CPU")
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        laf = torch.tensor([[4.0, 0.0, 16.0], [0.0, 4.0, 16.0]], dtype=dtype).view(1, 1, 2, 3)
        patches = kornia.feature.extract_patches_simple(img, laf, 8)
        assert patches.device == img.device
        self.assert_close(patches, kornia.feature.extract_patches_simple(img, laf.to(device), 8))

    def test_dynamo(self, device, dtype, torch_optimizer):
        laf = torch.rand(2, 4, 2, 3, device=device, dtype=dtype)
        img = torch.rand(2, 1, 40, 30, device=device, dtype=dtype)
        expected = kornia.feature.extract_patches_simple(img, laf, 10)
        op = torch_optimizer(kornia.feature.extract_patches_simple)
        self.assert_close(op(img, laf, 10), expected)
        # The fixture compiles without `fullgraph`, so it would pass on a graph break too, and the
        # extractor is meant to trace as one graph -- assert that directly.
        torch._dynamo.reset()
        self.assert_close(torch.compile(kornia.feature.extract_patches_simple, fullgraph=True)(img, laf, 10), expected)


class TestExtractPatchesPyr(BaseTester):
    def test_shape(self, device):
        laf = torch.rand(5, 4, 2, 3, device=device)
        img = torch.rand(5, 3, 100, 30, device=device)
        PS = 10
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    def test_non_zero(self, device):
        img = torch.zeros(1, 1, 24, 24, device=device)
        img[:, :, 10:, 20:] = 1.0
        laf = torch.tensor([[8.0, 0, 14.0], [0, 8.0, 8.0]], device=device).reshape(1, 1, 2, 3)

        PS = 32
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.mean().item() > 0.01
        assert patches.shape == (1, 1, 1, PS, PS)

    def test_same_odd(self, device, dtype):
        img = torch.arange(5)[None].repeat(5, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[2.0, 0, 2.0], [0, 2.0, 2.0]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_from_pyramid(img, laf, 5, 1.0)
        self.assert_close(img, patch[0])

    def test_same_even(self, device, dtype):
        img = torch.arange(4)[None].repeat(4, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[1.5, 0, 1.5], [0, 1.5, 1.5]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_from_pyramid(img, laf, 4, 1.0)
        self.assert_close(img, patch[0])

    def test_small_image_single_level(self, device, dtype):
        # When min(H, W) < 2 * PS, the pyramid cannot descend beyond level 0.
        # Sampling that image directly must retain the plain extractor's border behavior without
        # allocating a one-level atlas.
        PS = 16
        img = torch.rand(1, 1, 24, 24, device=device, dtype=dtype)  # 24 < 2*16=32 → only level 0
        laf = torch.tensor([[6.0, 0.0, 2.0], [0.0, 6.0, 2.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.shape == (1, 1, 1, PS, PS)
        assert patches.abs().sum().item() > 0
        self.assert_close(patches, kornia.feature.extract_patches_simple(img, laf, PS))

    def test_exception(self, device, dtype):
        # Batch disagreement is rejected at the boundary with a clear message instead of an
        # internal broadcasting error or a silently truncated result.
        img = torch.rand(3, 1, 32, 32, device=device, dtype=dtype)
        with pytest.raises(Exception, match="same batch size"):
            kornia.feature.extract_patches_from_pyramid(img, torch.rand(1, 2, 2, 3, device=device, dtype=dtype), 8)

    def test_mixed_dtype_laf_under_autocast(self, device):
        if device.type not in ("cpu", "cuda"):
            pytest.skip("autocast coverage is provided by CPU and CUDA")
        laf_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        img = torch.rand(1, 1, 64, 64, device=device)
        laf = torch.tensor([[8.0, 0.0, 32.0], [0.0, 8.0, 32.0]], device=device, dtype=laf_dtype).view(1, 1, 2, 3)
        with torch.autocast(device_type=device.type, dtype=laf_dtype):
            actual = kornia.feature.extract_patches_from_pyramid(img, laf, 8)
            expected = kornia.feature.extract_patches_from_pyramid(img, laf.float(), 8)
        assert actual.dtype == img.dtype
        self.assert_close(actual, expected)

    def test_mixed_dtype_preserves_laf_precision(self, device):
        if device.type != "cpu":
            pytest.skip("CPU provides deterministic reduced-precision reference coverage")
        img = torch.rand(1, 1, 64, 64).to(torch.float16)
        laf = torch.tensor([[8.1234, 0.2712, 31.789], [-0.1923, 7.9876, 32.321]]).view(1, 1, 2, 3)
        actual = kornia.feature.extract_patches_from_pyramid(img, laf, 8)
        expected = kornia.feature.extract_patches_from_pyramid(img.float(), laf, 8).to(img.dtype)
        assert actual.dtype == img.dtype
        assert torch.equal(actual, expected)

    def test_one_pixel_axis_does_not_raise(self, device, dtype):
        # A 1-pixel image axis has no spatial extent. `normalize_laf`/`denormalize_laf` count it as
        # one pixel instead of dividing by zero, and the levelwise sampler uses a zero grid scale
        # for that axis, so the only real pixel is repeated instead of leaking a NaN. Both the
        # default pixel-LAF entry point and the pre-normalized one stay finite, on both extractors.
        for shape in ((1, 1, 5, 1), (1, 1, 1, 5)):
            img = torch.rand(shape, device=device, dtype=dtype)
            laf = torch.rand(1, 1, 2, 3, device=device, dtype=dtype)
            for normalize_lafs in (False, True):
                for extract in (
                    kornia.feature.extract_patches_from_pyramid,
                    kornia.feature.extract_patches_simple,
                ):
                    patches = extract(img, laf, 4, normalize_lafs)
                    assert patches.shape == (1, 1, 1, 4, 4)
                    assert bool(patches.isfinite().all())

    def test_one_pixel_axis_preserves_non_singleton_extent(self, device, dtype):
        # A singleton axis must collapse only itself. The other axis still has spatial extent, so
        # a full-image LAF over a 1x5 or 5x1 ramp must retain that ramp instead of degenerating to
        # the center pixel in both directions.
        for h, w in ((1, 5), (5, 1)):
            img = torch.arange(5, device=device, dtype=dtype).reshape(1, 1, h, w)
            pixel_laf = torch.tensor(
                [
                    [float(max(w - 1, 1)) / 2.0, 0.0, float(w - 1) / 2.0],
                    [0.0, float(max(h - 1, 1)) / 2.0, float(h - 1) / 2.0],
                ],
                device=device,
                dtype=dtype,
            ).view(1, 1, 2, 3)
            normalized_laf = kornia.feature.normalize_laf(pixel_laf, img)
            expected = img.expand(1, 1, 5, 5).unsqueeze(1)

            for laf, normalize_lafs in ((pixel_laf, True), (normalized_laf, False)):
                simple = kornia.feature.extract_patches_simple(img, laf, 5, normalize_lafs)
                pyramid = kornia.feature.extract_patches_from_pyramid(img, laf, 5, normalize_lafs)
                self.assert_close(simple, expected)
                self.assert_close(pyramid, expected)

    @pytest.mark.parametrize("height,width", [(12, 12), (8, 8), (2, 8), (8, 2)])
    def test_one_pixel_level_does_not_raise(self, device, dtype, height, width):
        # PS=1 can make a pyramid descend to a 1-pixel level. A 12-pixel axis reaches it safely
        # through 12 -> 6 -> 3 -> 1, while a power-of-two or rectangular axis reaches 2 first;
        # `pyrdown` cannot reflect-pad that 2-pixel source, so it must remain the coarsest usable
        # level instead of attempting the final downsample.
        img = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        laf = torch.tensor([[3.0, 0.0, width / 2.0], [0.0, 3.0, height / 2.0]], device=device, dtype=dtype).view(
            1, 1, 2, 3
        )
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, 1)
        assert patches.shape == (1, 1, 1, 1, 1)
        assert bool(patches.isfinite().all())

    def test_nonfinite_laf_returns_zero_patch_with_safe_backward(self, device, dtype, monkeypatch):
        # A training-time detector can emit a non-finite LAF frame. Passing a NaN grid to the CPU
        # border-padding backward kernel can segfault the process, even when forward output is
        # zeroed afterwards. Both paths must sanitize the whole frame before sampling, including
        # when only its center is invalid, and return zero output/gradient without changing finite
        # frames. Infinity is invalid for the same API-level reason, independent of backend quirks.
        import kornia.feature.laf as laf_module

        PS = 8
        img = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        nan = float("nan")
        inf = float("inf")
        laf = torch.tensor(
            [
                [[8.0, 0.0, 32.0], [0.0, 8.0, 32.0]],
                [[nan, nan, nan], [nan, nan, nan]],
                [[8.0, 0.0, nan], [0.0, 8.0, nan]],
                [[8.0, 0.0, inf], [0.0, 8.0, inf]],
            ],
            device=device,
            dtype=dtype,
        ).view(1, 4, 2, 3)

        expected_finite = kornia.feature.extract_patches_from_pyramid(img, laf[:, :1], PS)
        for atlas_fits in (True, False):
            monkeypatch.setattr(laf_module, "_pyramid_atlas_fits", lambda *args, fits=atlas_fits: fits)
            grad_img = img.detach().clone().requires_grad_()
            grad_laf = laf.detach().clone().requires_grad_()
            patches = kornia.feature.extract_patches_from_pyramid(grad_img, grad_laf, PS)
            self.assert_close(patches[:, :1], expected_finite)
            assert patches[:, 1:].abs().sum().item() == 0
            patches.sum().backward()
            assert grad_img.grad is not None and bool(grad_img.grad.isfinite().all())
            assert grad_laf.grad is not None and bool(grad_laf.grad.isfinite().all())
            assert grad_laf.grad[:, 1:].abs().sum().item() == 0

    def test_giant_laf_uses_actual_coarsest_level(self, device, dtype):
        # The nominal level for this LAF is beyond the levels that can provide a PS-sized
        # patch. It must sample the actual coarsest level rather than becoming zero padding.
        PS = 16
        img = torch.arange(128 * 128, device=device, dtype=dtype).reshape(1, 1, 128, 128)
        # The LAF is deliberately float32: 1e5 overflows float16 to infinity, which the non-finite
        # contract turns into a zero patch, so a half LAF would test the opposite behavior. Paired
        # with the injected image dtype it keeps the giant *finite* case under test on every dtype
        # and exercises the mixed-dtype promotion path at the same time.
        laf = torch.tensor([[1.0e5, 0.0, 64.0], [0.0, 1.0e5, 64.0]], device=device, dtype=torch.float32).view(
            1, 1, 2, 3
        )

        actual = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        nlaf = kornia.feature.normalize_laf(laf, img)
        # Build the reference pyramid the way the extractor does for half dtypes -- in float32 --
        # so the reference itself does not hit the missing half reflect-pad kernel on old torch.
        coarsest = img.float() if dtype in (torch.float16, torch.bfloat16) else img
        for _ in range(3):
            coarsest = kornia.geometry.transform.pyrdown(coarsest)
        expected = kornia.feature.extract_patches_simple(coarsest.to(dtype), nlaf, PS, False)

        self.assert_close(actual, expected)
        assert actual.abs().sum().item() > 0

    def test_oversized_atlas_uses_equivalent_levelwise_fallback(self, device, dtype, monkeypatch):
        import kornia.feature.laf as laf_module

        PS = 16
        img = torch.rand(1, 1, 65, 97, device=device, dtype=dtype)
        laf = torch.tensor(
            [[[4.0, 0.0, 48.0], [0.0, 4.0, 32.0]], [[24.0, 0.0, 48.0], [0.0, 24.0, 32.0]]],
            device=device,
            dtype=dtype,
        ).view(1, 2, 2, 3)

        monkeypatch.setattr(laf_module, "_pyramid_atlas_fits", lambda *args: False)
        fallback = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        monkeypatch.setattr(laf_module, "_pyramid_atlas_fits", lambda *args: True)
        atlas = kornia.feature.extract_patches_from_pyramid(img, laf, PS)

        self.assert_close(fallback, atlas)

    def test_atlas_guards_preserve_level_border_value_and_gradient(self, device, dtype, monkeypatch):
        import kornia.feature.laf as laf_module

        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the coordinate-rounding leak this test pins is below half-precision resolution")

        def sample(limit, size, patch_size):
            monkeypatch.setattr(laf_module, "_pyramid_atlas_fits", lambda *args: limit > 0)
            img = torch.zeros(1, 1, size, size, device=device, dtype=dtype)
            img[:, :, :, -1] = 1.0
            # A quarter pixel OUTSIDE the outermost pixel center, so the clamp engages strictly on
            # every backend: probing exactly at the center can round onto the clamp bound, where
            # the subgradient convention differs between CPU and MPS.
            center_x = 1.0 - 1.0 / (4.0 * size)
            nlaf = torch.tensor(
                [[[[0.0, 0.0, center_x], [0.0, 0.0, 0.5]]]], device=device, dtype=dtype, requires_grad=True
            )
            patch = kornia.feature.extract_patches_from_pyramid(img, nlaf, patch_size, False)
            value = patch[0, 0, 0, patch_size // 2, patch_size // 2]
            grad = torch.autograd.grad(value, nlaf)[0][0, 0, 0, 2]
            return patch, grad

        # 2495 exposes a forward-coordinate rounding leak without guards; 32 exposes the
        # non-zero outward center gradient even when the rounded forward value happens to match.
        # The discriminating pin is the patch CENTER: its sampling coordinate is clamped to the
        # border-pixel center, so with the guard both bilinear partners are exactly the border
        # value and the sample stays within a couple of float32 ulp of 1.0, while a missing guard
        # blends in the neighbouring level at the coordinate-rounding scale (~5e-5 in float32 at
        # size 2495, measured). Interior pixels legitimately differ between the atlas and
        # levelwise paths at that same rounding scale -- and across platforms -- so the
        # whole-patch comparison uses the standard tolerances instead of `torch.equal`.
        for size, patch_size in ((2495, 8), (32, 5)):
            fallback, fallback_grad = sample(0, size, patch_size)
            atlas, atlas_grad = sample(1 << 60, size, patch_size)
            self.assert_close(atlas, fallback)
            center = patch_size // 2
            assert abs(atlas[0, 0, 0, center, center].item() - 1.0) < 1e-5
            assert abs(fallback[0, 0, 0, center, center].item() - 1.0) < 1e-5
            assert atlas_grad == fallback_grad == 0.0

    def test_multi_level_uses_correct_pyramid_level(self, device, dtype):
        # Two LAFs with very different scales should be extracted from different pyramid levels.
        # We verify the output shape and that the function runs without errors.
        PS = 8
        img = torch.rand(1, 1, 128, 128, device=device, dtype=dtype)
        # Small-scale LAF (extracted at level 0) and large-scale LAF (extracted at higher level).
        laf_small = torch.tensor([[2.0, 0.0, 64.0], [0.0, 2.0, 64.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        laf_large = torch.tensor([[32.0, 0.0, 64.0], [0.0, 32.0, 64.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        laf_both = torch.cat([laf_small, laf_large], dim=1)
        patches = kornia.feature.extract_patches_from_pyramid(img, laf_both, PS)
        assert patches.shape == (1, 2, 1, PS, PS)

    def test_gradcheck(self, device):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]], device=device, dtype=torch.float64)
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 20, 30, device=device, dtype=torch.float64)
        PS = 11
        self.gradcheck(
            kornia.feature.extract_patches_from_pyramid,
            (img, nlaf, PS, False),
            nondet_tol=1e-8,
        )

    def test_gradcheck_chunked(self, device, monkeypatch):
        # Gradients must survive the chunked sampling loop and its slice writes, on both the
        # atlas path and the levelwise fallback.
        import kornia.feature.laf as laf_module

        monkeypatch.setattr(laf_module, "_grid_chunk_lafs", lambda *args: 1)
        nlaf = torch.tensor(
            [[[0.1, 0.001, 0.4], [0.0, 0.1, 0.5]], [[0.05, 0.0, 0.6], [0.0, 0.05, 0.4]]],
            device=device,
            dtype=torch.float64,
        ).view(1, 2, 2, 3)
        img = torch.rand(1, 2, 20, 30, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.extract_patches_from_pyramid, (img, nlaf, 7, False), nondet_tol=1e-8)
        monkeypatch.setattr(laf_module, "_pyramid_atlas_fits", lambda *args: False)
        self.gradcheck(kornia.feature.extract_patches_from_pyramid, (img, nlaf, 7, False), nondet_tol=1e-8)

    def test_batch_independence(self, device, dtype):
        # Each patch must come only from its own batch element, across pyramid levels: the two
        # scales route the patches through level 0 and a downsampled level respectively.
        B, PS = 3, 8
        img = torch.arange(B, device=device, dtype=dtype).view(B, 1, 1, 1).expand(B, 1, 64, 64).contiguous()
        laf_small = torch.tensor([[4.0, 0.0, 32.0], [0.0, 4.0, 32.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        laf_large = torch.tensor([[16.0, 0.0, 32.0], [0.0, 16.0, 32.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        laf = torch.cat([laf_small, laf_large], dim=1).expand(B, 2, 2, 3)
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.shape == (B, 2, 1, PS, PS)
        expected = torch.arange(B, device=device, dtype=dtype).view(B, 1, 1, 1, 1).expand(B, 2, 1, PS, PS)
        self.assert_close(patches, expected)

    def test_empty(self, device, dtype):
        # Degenerate shapes follow the empty-in -> empty-out convention: the batched rewrite must
        # not call `affine_grid` on an empty batch (main returned an empty result for B=0).
        PS = 8
        patches = kornia.feature.extract_patches_from_pyramid(
            torch.rand(0, 2, 32, 32, device=device, dtype=dtype), torch.rand(0, 3, 2, 3, device=device, dtype=dtype), PS
        )
        assert patches.shape == (0, 3, 2, PS, PS)
        assert patches.dtype == dtype
        patches = kornia.feature.extract_patches_from_pyramid(
            torch.rand(2, 1, 32, 32, device=device, dtype=dtype), torch.rand(2, 0, 2, 3, device=device, dtype=dtype), PS
        )
        assert patches.shape == (2, 0, 1, PS, PS)

    def test_chunked_matches_single_call(self, device, dtype, monkeypatch):
        # Forcing one LAF per chunk must reproduce the single-call result on both the atlas path
        # and the levelwise fallback, since chunking only splits the grid along N.
        import kornia.feature.laf as laf_module

        torch.manual_seed(0)
        PS = 8
        img = torch.rand(2, 3, 64, 64, device=device, dtype=dtype)
        laf = torch.zeros(2, 5, 2, 3, device=device, dtype=dtype)
        laf[..., 0, 0] = laf[..., 1, 1] = torch.tensor([2.0, 4.0, 9.0, 17.0, 33.0], device=device, dtype=dtype)
        laf[..., :, 2] = torch.rand(2, 5, 2, device=device, dtype=dtype) * 40 + 12
        default_chunk = laf_module._grid_chunk_lafs
        expected = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        monkeypatch.setattr(laf_module, "_grid_chunk_lafs", lambda *args: 1)
        chunked = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        self.assert_close(chunked, expected)
        if device.type == "cpu":
            assert torch.equal(chunked, expected)
        monkeypatch.setattr(laf_module, "_pyramid_atlas_fits", lambda *args: False)
        fallback_chunked = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        monkeypatch.setattr(laf_module, "_grid_chunk_lafs", default_chunk)
        fallback_expected = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        self.assert_close(fallback_chunked, fallback_expected)
        if device.type == "cpu":
            assert torch.equal(fallback_chunked, fallback_expected)

    def test_channel_laf_correspondence(self, device, dtype):
        # The patch at [b, n] must equal the one extracted for that LAF alone. At ch=1 or N=1 a
        # scrambled unfold is bit-identical to the right one (it permutes a size-1 dim), so this
        # pins ch>1 together with N>1, across pyramid levels.
        B, N, ch, PS = 2, 4, 3, 8
        torch.manual_seed(0)
        img = torch.rand(B, ch, 64, 64, device=device, dtype=dtype)
        laf = torch.zeros(B, N, 2, 3, device=device, dtype=dtype)
        laf[..., 0, 0] = laf[..., 1, 1] = torch.tensor([3.0, 6.0, 17.0, 33.0], device=device, dtype=dtype)
        laf[..., :, 2] = torch.rand(B, N, 2, device=device, dtype=dtype) * 40 + 12
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        for b in range(B):
            for n in range(N):
                single = kornia.feature.extract_patches_from_pyramid(img[b : b + 1], laf[b : b + 1, n : n + 1], PS)
                self.assert_close(patches[b, n], single[0, 0])

    @pytest.mark.parametrize("half_dtype", [torch.float16, torch.bfloat16])
    def test_border_patches_stay_in_range(self, device, half_dtype):
        # Same border read as in `TestExtractPatchesSimple`, at pyramid level 0: `_corner_border_laf`
        # has a scale below the patch size, so the patch comes from the undownsampled image and the
        # level-0 result must equal the plain extractor's byte for byte. A larger LAF would be served
        # by a blurred level whose near-constant patch stays in range even when the read is wrong.
        # The whole pyramid (pad, `pyrdown`, sampling) runs in float32 for half inputs, so no
        # reduced-precision pad or `grid_sample` kernel gate is needed on any torch version.
        if device.type != "cpu":
            pytest.skip(_HALF_BORDER_SKIP)
        torch.manual_seed(0)
        img = torch.rand(1, 1, 256, 256, device=device).to(half_dtype)
        laf = _corner_border_laf(device, half_dtype)
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, 16)
        assert patches.dtype == half_dtype
        assert bool(patches.isfinite().all())
        assert patches.min() >= img.min()
        assert patches.max() <= img.max()
        self.assert_close(patches, kornia.feature.extract_patches_simple(img, laf, 16))

    def test_reduced_precision_level_zero_matches_float_reference(self, device, dtype):
        # Unlike the explicit half_dtype regression above, this uses the global dtype fixture so
        # CUDA half tests run in the suite's per-test subprocess isolation. The float32 grid policy
        # is universal even though the out-of-bounds kernel bug that motivated it is CPU-specific.
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("reduced-precision sampling policy")
        img = torch.rand(1, 1, 256, 256, device=device).to(dtype)
        laf = _corner_border_laf(device, dtype)
        simple = kornia.feature.extract_patches_simple(img, laf, 16)
        pyramid = kornia.feature.extract_patches_from_pyramid(img, laf, 16)
        reference = kornia.feature.extract_patches_simple(img.float(), laf.float(), 16).to(dtype)
        assert simple.dtype == pyramid.dtype == dtype
        assert bool(simple.isfinite().all()) and bool(pyramid.isfinite().all())
        self.assert_close(simple, reference)
        self.assert_close(pyramid, reference)

    def test_laf_on_another_device(self, device, dtype, monkeypatch):
        # The small LAF is moved to the image before grid construction, so the pyramid extractor
        # accepts a cross-device pair like `extract_patches_simple` -- on the atlas path AND on the
        # levelwise fallback, which a 64x64 image never reaches on its own (the contract must not
        # flip with the image size).
        import kornia.feature.laf as laf_module

        if device.type == "cpu":
            pytest.skip("needs a second device besides the CPU")
        img = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        laf = torch.tensor([[8.0, 0.0, 32.0], [0.0, 8.0, 32.0]], dtype=dtype).view(1, 1, 2, 3)
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, 8)
        assert patches.device == img.device
        self.assert_close(patches, kornia.feature.extract_patches_from_pyramid(img, laf.to(device), 8))
        monkeypatch.setattr(laf_module, "_pyramid_atlas_fits", lambda *args: False)
        fallback = kornia.feature.extract_patches_from_pyramid(img, laf, 8)
        assert fallback.device == img.device
        self.assert_close(fallback, kornia.feature.extract_patches_from_pyramid(img, laf.to(device), 8))

    def test_dynamo(self, device, dtype, torch_optimizer, optimizer_backend):
        if optimizer_backend == "jit":
            # `pyrdown` -> `filter2d` closes over a `set` of valid border modes, which TorchScript
            # cannot compile; unrelated to this extractor and to the compile path CI exercises.
            pytest.skip("`extract_patches_from_pyramid` reaches a non-scriptable `filter2d`")
        laf = torch.rand(2, 4, 2, 3, device=device, dtype=dtype)
        img = torch.rand(2, 1, 40, 30, device=device, dtype=dtype)
        expected = kornia.feature.extract_patches_from_pyramid(img, laf, 10)
        op = torch_optimizer(kornia.feature.extract_patches_from_pyramid)
        self.assert_close(op(img, laf, 10), expected)
        # The fixture compiles without `fullgraph`, so it would pass on a graph break too, and the
        # extractor is meant to trace as one graph -- assert that directly.
        torch._dynamo.reset()
        compiled = torch.compile(kornia.feature.extract_patches_from_pyramid, fullgraph=True)
        self.assert_close(compiled(img, laf, 10), expected)

    def test_dynamo_levelwise_fallback(self, device, dtype, torch_optimizer, optimizer_backend, monkeypatch):
        import kornia.feature.laf as laf_module

        if optimizer_backend == "jit":
            # Same TorchScript incompatibility `test_dynamo` skips.
            pytest.skip("`extract_patches_from_pyramid` reaches a non-scriptable `filter2d`")
        monkeypatch.setattr(laf_module, "_pyramid_atlas_fits", lambda *args: False)
        laf = torch.rand(2, 4, 2, 3, device=device, dtype=dtype)
        img = torch.rand(2, 1, 65, 97, device=device, dtype=dtype)
        expected = kornia.feature.extract_patches_from_pyramid(img, laf, 16)
        op = torch_optimizer(kornia.feature.extract_patches_from_pyramid)
        self.assert_close(op(img, laf, 16), expected)
        torch._dynamo.reset()
        compiled = torch.compile(kornia.feature.extract_patches_from_pyramid, fullgraph=True)
        self.assert_close(compiled(img, laf, 16), expected)

    def test_dynamo_chunked(self, device, dtype, torch_optimizer, optimizer_backend, monkeypatch):
        # The chunk loop unrolls at trace time for static shapes; forcing one LAF per chunk
        # exercises it.
        import kornia.feature.laf as laf_module

        if optimizer_backend == "jit":
            # Same TorchScript incompatibility `test_dynamo` skips.
            pytest.skip("`extract_patches_from_pyramid` reaches a non-scriptable `filter2d`")
        monkeypatch.setattr(laf_module, "_grid_chunk_lafs", lambda *args: 1)
        laf = torch.rand(2, 4, 2, 3, device=device, dtype=dtype)
        img = torch.rand(2, 1, 65, 97, device=device, dtype=dtype)
        expected = kornia.feature.extract_patches_from_pyramid(img, laf, 16)
        op = torch_optimizer(kornia.feature.extract_patches_from_pyramid)
        self.assert_close(op(img, laf, 16), expected)
        torch._dynamo.reset()
        compiled = torch.compile(kornia.feature.extract_patches_from_pyramid, fullgraph=True)
        self.assert_close(compiled(img, laf, 16), expected)


def test_nonfinite_laf_backward_does_not_crash_the_interpreter():
    """Backward through a non-finite LAF must not take the interpreter down.

    Both extractors advertise zero patches and zero gradients for a non-finite LAF frame. The
    failure mode of a regression is not a wrong number but a native segfault inside
    ``grid_sampler_2d_backward`` -- torch's CPU ``padding_mode="border"`` kernel on a NaN grid --
    which would kill the pytest process itself rather than fail a test. A fresh interpreter turns
    that into an ordinary assertion on the exit code. CPU-only on purpose: the crashing kernel is
    the CPU one, and the guard needs no device fixture to be meaningful.
    """
    script = textwrap.dedent(
        """
        import torch

        import kornia

        img = torch.rand(1, 1, 64, 64, requires_grad=True)
        # Only the center is invalid: the whole frame must still be treated as invalid.
        laf = torch.tensor([[[[8.0, 0.0, float("nan")], [0.0, 8.0, 32.0]]]])
        extractors = (kornia.feature.extract_patches_simple, kornia.feature.extract_patches_from_pyramid)
        for extract in extractors:
            patches = extract(img, laf, 8)
            assert patches.abs().sum().item() == 0.0, extract.__name__
            patches.sum().backward()
        assert img.grad is not None and bool(img.grad.isfinite().all())
        """
    )
    # Trusted, fixed command (the current interpreter running a literal script); no external input.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"non-finite LAF backward exited with {result.returncode}:\n{result.stdout}\n{result.stderr}"
    )


class TestLAFIsTouchingBoundary(BaseTester):
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        assert (5, 3) == kornia.feature.laf_is_inside_image(inp, img).shape

    def test_touch(self, device):
        w, h = 10, 5
        img = torch.rand(1, 3, h, w, device=device)
        laf = torch.tensor([[[[10, 0, 3], [0, 10, 3]], [[1, 0, 5], [0, 1, 2]]]], device=device).float()
        expected = torch.tensor([[False, True]], device=device)
        assert torch.all(kornia.feature.laf_is_inside_image(laf, img) == expected).item()

    @staticmethod
    def _radius_laf(cx, cy, radius, device, dtype):
        return kornia.feature.laf_from_center_scale_ori(
            torch.tensor([[[cx, cy]]], device=device, dtype=dtype),
            torch.full((1, 1, 1, 1), radius, device=device, dtype=dtype),
        )

    def test_boundary_is_the_last_valid_pixel(self, device, dtype):
        """Valid coordinates run 0 .. size-1, matching `get_laf_center`'s documented convention.

        A LAF reaching exactly `w - 1` is inside; anything past it is not (see #4064).
        """
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("boundary comparison needs full precision")
        w = h = 32  # valid coordinates 0 .. 31
        img = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
        r = 2.0

        # rightmost boundary lands exactly on 31.0 -> inside
        assert bool(kornia.feature.laf_is_inside_image(self._radius_laf(29.0, 16.0, r, device, dtype), img)[0, 0])
        # ... and half a pixel past it -> outside
        assert not bool(kornia.feature.laf_is_inside_image(self._radius_laf(29.5, 16.0, r, device, dtype), img)[0, 0])
        # same for the bottom edge
        assert bool(kornia.feature.laf_is_inside_image(self._radius_laf(16.0, 29.0, r, device, dtype), img)[0, 0])
        assert not bool(kornia.feature.laf_is_inside_image(self._radius_laf(16.0, 29.5, r, device, dtype), img)[0, 0])

    def test_boundary_is_symmetric(self, device, dtype):
        """The low and high edges must be equally strict."""
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("boundary comparison needs full precision")
        w = h = 32
        img = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
        r = 2.0
        # negative offsets push the LAF past the edge, which is where an asymmetric
        # upper bound shows up: on the low side it is rejected, on the high side it was not.
        for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
            low = kornia.feature.laf_is_inside_image(self._radius_laf(r + offset, 16.0, r, device, dtype), img)
            high = kornia.feature.laf_is_inside_image(
                self._radius_laf(float(w - 1) - r - offset, 16.0, r, device, dtype), img
            )
            assert bool(low[0, 0]) == bool(high[0, 0]), f"asymmetric at offset {offset}"

    def test_border_argument_shrinks_both_edges(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("boundary comparison needs full precision")
        w = h = 32
        img = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
        r = 2.0
        # with border=3 the usable range is 3 .. 28, so a radius-2 LAF centred at 26 just fits
        assert bool(kornia.feature.laf_is_inside_image(self._radius_laf(26.0, 16.0, r, device, dtype), img, 3)[0, 0])
        assert not bool(
            kornia.feature.laf_is_inside_image(self._radius_laf(26.5, 16.0, r, device, dtype), img, 3)[0, 0]
        )

    def test_jit(self, device, dtype):
        w, h = 10, 5
        img = torch.rand(1, 3, h, w, device=device)
        laf = torch.tensor([[[[10, 0, 3], [0, 10, 3]], [[1, 0, 5], [0, 1, 2]]]], device=device).float()
        model = kornia.feature.laf_is_inside_image
        model_jit = torch.jit.script(kornia.feature.laf_is_inside_image)
        self.assert_close(model(laf, img), model_jit(laf, img))


class TestGetCreateLAF(BaseTester):
    def test_shape(self, device):
        xy = torch.ones(1, 3, 2, device=device)
        ori = torch.ones(1, 3, 1, device=device)
        scale = torch.ones(1, 3, 1, 1, device=device)
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        assert laf.shape == (1, 3, 2, 3)

    def test_laf(self, device):
        xy = torch.ones(1, 1, 2, device=device)
        ori = torch.zeros(1, 1, 1, device=device)
        scale = 5 * torch.ones(1, 1, 1, 1, device=device)
        expected = torch.tensor([[[[5, 0, 1], [0, 5, 1]]]], device=device).float()
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        self.assert_close(laf, expected)

    def test_laf_def(self, device):
        xy = torch.ones(1, 1, 2, device=device)
        expected = torch.tensor([[[[1, 0, 1], [0, 1, 1]]]], device=device).float()
        laf = kornia.feature.laf_from_center_scale_ori(xy)
        self.assert_close(laf, expected)

    def test_cross_consistency(self, device):
        batch_size, channels = 3, 2
        xy = torch.rand(batch_size, channels, 2, device=device)
        ori = torch.rand(batch_size, channels, 1, device=device)
        scale = torch.abs(torch.rand(batch_size, channels, 1, 1, device=device))
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        scale2 = kornia.feature.get_laf_scale(laf)
        self.assert_close(scale, scale2)
        xy2 = kornia.feature.get_laf_center(laf)
        self.assert_close(xy2, xy)
        ori2 = kornia.feature.get_laf_orientation(laf)
        self.assert_close(ori2, ori)

    def test_gradcheck(self, device):
        batch_size, channels = 3, 2
        xy = torch.rand(batch_size, channels, 2, device=device, dtype=torch.float64)
        ori = torch.rand(batch_size, channels, 1, device=device, dtype=torch.float64)
        scale = torch.abs(torch.rand(batch_size, channels, 1, 1, device=device, dtype=torch.float64))
        self.gradcheck(kornia.feature.laf_from_center_scale_ori, (xy, scale, ori))

    @pytest.mark.skip("Depends on angle-to-rotation-matric")
    def test_jit(self, device, dtype):
        batch_size, channels = 3, 2
        xy = torch.rand(batch_size, channels, 2, device=device)
        ori = torch.rand(batch_size, channels, 1, device=device)
        scale = torch.abs(torch.rand(batch_size, channels, 1, 1, device=device))
        model = kornia.feature.laf_from_center_scale_ori
        model_jit = torch.jit.script(kornia.feature.laf_from_center_scale_ori)
        self.assert_close(model(xy, scale, ori), model_jit(xy, scale, ori))


class TestGetLAF3pts(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        out = kornia.feature.laf_to_three_points(inp)
        assert out.shape == inp.shape

    def test_batch_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        out = kornia.feature.laf_to_three_points(inp)
        assert out.shape == inp.shape

    def test_conversion(self, device):
        inp = torch.tensor([[1, 0, 2], [0, 1, 3]], device=device).float().view(1, 1, 2, 3)
        expected = torch.tensor([[3, 2, 2], [3, 4, 3]], device=device).float().view(1, 1, 2, 3)
        threepts = kornia.feature.laf_to_three_points(inp)
        self.assert_close(threepts, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.laf_to_three_points, (inp,))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.laf_to_three_points
        model_jit = torch.jit.script(kornia.feature.laf_to_three_points)
        self.assert_close(model(inp), model_jit(inp))


class TestGetLAFFrom3pts(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        out = kornia.feature.laf_from_three_points(inp)
        assert out.shape == inp.shape

    def test_batch_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        out = kornia.feature.laf_from_three_points(inp)
        assert out.shape == inp.shape

    def test_conversion(self, device):
        expected = torch.tensor([[1, 0, 2], [0, 1, 3]], device=device).float().view(1, 1, 2, 3)
        inp = torch.tensor([[3, 2, 2], [3, 4, 3]], device=device).float().view(1, 1, 2, 3)
        threepts = kornia.feature.laf_from_three_points(inp)
        self.assert_close(threepts, expected)

    def test_cross_consistency(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        inp_2 = kornia.feature.laf_from_three_points(inp)
        inp_2 = kornia.feature.laf_to_three_points(inp_2)
        self.assert_close(inp_2, inp)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.laf_from_three_points, (inp,))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.laf_from_three_points
        model_jit = torch.jit.script(kornia.feature.laf_from_three_points)
        self.assert_close(model(inp), model_jit(inp))


class TestTransformLAFs(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_points", [2, 3, 5])
    def test_transform_points(self, batch_size, num_points, device, dtype):
        laf_bank = torch.tensor(
            [[[0.4, 0.1, 1.0], [-0.2, 0.3, 2.0]], [[-0.3, 0.2, -1.0], [0.1, 0.5, 0.5]]],
            device=device,
            dtype=dtype,
        )
        homography_bank = torch.tensor(
            [
                [[1.2, 0.1, 0.5], [-0.2, 0.9, 0.3], [0.02, -0.01, 1.0]],
                [[0.8, -0.15, -0.2], [0.05, 1.1, 0.4], [-0.015, 0.025, 1.0]],
            ],
            device=device,
            dtype=dtype,
        )
        # Closed-form projection of the two LAFs by the two homographies, generated in float64:
        # p' = (H[:2, :2] @ p + H[:2, 2]) / (H[2, :2] @ p + H[2, 2]).
        expected_bank = torch.tensor(
            [
                [
                    [[0.4366336634, 0.1520520521, 1.9], [-0.2762376238, 0.2521521522, 1.9]],
                    [
                        [-0.3663911846, 0.2970568104, -0.6666666667],
                        [0.1620046620, 0.4219449271, 0.9743589744],
                    ],
                ],
                [
                    [
                        [0.3449105525, 0.0319508833, 0.2898550725],
                        [-0.1678083484, 0.3070486851, 2.5603864734],
                    ],
                    [
                        [-0.2394165288, 0.0915517577, -1.0462287105],
                        [0.0859048943, 0.5319950165, 0.8759124088],
                    ],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        batch_indices = torch.arange(batch_size, device=device) % len(homography_bank)
        point_indices = torch.arange(num_points, device=device) % len(laf_bank)
        lafs_src = laf_bank[point_indices].unsqueeze(0).expand(batch_size, -1, -1, -1)
        dst_homo_src = homography_bank[batch_indices]
        expected = expected_bank[batch_indices[:, None], point_indices[None, :]]

        actual = kornia.feature.perspective_transform_lafs(dst_homo_src, lafs_src)

        half_tolerance = 3 * torch.finfo(dtype).eps if dtype in (torch.float16, torch.bfloat16) else None
        self.assert_close(actual, expected, atol=half_tolerance, rtol=half_tolerance)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points = 2, 3
        eye_size = 3
        points_src = torch.rand(batch_size, num_points, 2, 3, device=device, dtype=torch.float64)
        dst_homo_src = create_random_homography(points_src, eye_size)
        # evaluate function gradient
        self.gradcheck(kornia.feature.perspective_transform_lafs, (dst_homo_src, points_src))
