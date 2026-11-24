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


import pytest
import torch

from kornia.geometry.transform import find_similarity_transform3d

from testing.base import BaseTester


class TestFindSimilarityTransform3d(BaseTester):
    @pytest.mark.parametrize("estimate_scale", [False, True])
    def test_convention_source_to_destination(self, device, dtype, estimate_scale):
        # (x,y,z) -> (-y,x,z), then a uniform scale and translation.
        # The second cloud is planar and still fixes a unique 3D rotation.
        src = torch.tensor(
            [[[1, 2, 3], [-1, 0, 3], [4, 1, 4], [0, 1, 3]], [[1, 2, 0], [-1, 0, 0], [4, 1, 0], [0, 1, 0]]],
            device=device,
            dtype=dtype,
        )
        rotation = src.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).expand(2, 3, 3)
        scale = src.new_tensor([2, 3] if estimate_scale else [1, 1])
        translation = src.new_tensor([[2, -3, 1], [-4, 8, 0]])
        dst = scale[:, None, None] * torch.stack((-src[..., 1], src[..., 0], src[..., 2]), dim=-1)
        dst = dst + translation[:, None]
        fit = find_similarity_transform3d(src, dst, estimate_scale=estimate_scale)
        assert fit.valid.tolist() == [True, True]
        self.assert_close(fit.rotation, rotation)
        self.assert_close(fit.scale, scale)
        self.assert_close(fit.translation, translation)

    def test_convention_rejects_reflection(self, device, dtype):
        # The cross covariance of these centred axis points has diagonal
        # (-9, 4, 1) / 3. The optimal proper rotation reverses x and z,
        # and the fitted scale is (9 + 4 - 1) / (9 + 4 + 1).
        src = torch.tensor(
            [[[3, 0, 0], [-3, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 1], [0, 0, -1]]], device=device, dtype=dtype
        )
        dst = src * src.new_tensor([-1, 1, 1])
        fit = find_similarity_transform3d(src, dst)
        assert fit.valid.item()
        self.assert_close(fit.rotation, src.new_tensor([[[-1, 0, 0], [0, 1, 0], [0, 0, -1]]]))
        self.assert_close(fit.translation, src.new_zeros((1, 3)))
        self.assert_close(fit.scale, src.new_tensor([6 / 7]))

    def test_zero_weight_and_integer_multiplicity(self, device, dtype):
        src = torch.tensor([[[10, -3, 7], [0, 0, 0], [2, 1, 0], [0, 2, 1], [1, 0, 2]]], device=device, dtype=dtype)
        dst = src.new_tensor([[[-7, 50, 3], [1, -2, 3], [5, 0, 3], [1, 3, 7], [3, -2, 8]]])
        fit = find_similarity_transform3d(src, dst, src.new_tensor([[0, 1, 2, 1, 1]]))
        indices = torch.tensor([1, 2, 2, 3, 4], device=device)
        repeated = find_similarity_transform3d(src[:, indices], dst[:, indices])
        assert fit.valid.item()
        assert repeated.valid.item()
        self.assert_close(fit.rotation, repeated.rotation)
        self.assert_close(fit.translation, repeated.translation)
        self.assert_close(fit.scale, repeated.scale)

    def test_invalid_lanes_give_finite_zero_gradients(self, device, dtype):
        x = torch.tensor(
            [[[1, 0, 0], [2, 0, 0], [3, 0, 0]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        y = (x.detach() * 2 + 1).requires_grad_()
        weights = x.new_tensor([[1, 1, 1], [0, 0, 0]], requires_grad=True)
        fit = find_similarity_transform3d(x, y, weights)
        assert not fit.valid.any()
        self.assert_close(fit.rotation, torch.eye(3, device=device, dtype=dtype).expand(2, 3, 3))
        self.assert_close(fit.translation, x.new_zeros((2, 3)))
        self.assert_close(fit.scale, x.new_ones((2,)))
        total = fit.rotation.sum() + fit.translation.sum() + fit.scale.sum()
        for grad in torch.autograd.grad(total, (x, y, weights)):
            self.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=0)

    def test_repeated_lower_eigenvalues_gradients(self, device, dtype):
        src = torch.cat((torch.eye(3, device=device, dtype=dtype), -torch.eye(3, device=device, dtype=dtype)))[None]
        dst = src * 2 + src.new_tensor([1, -2, 3])
        weights = src.new_ones(src.shape[:2])

        def aligned(x, y, w):
            fit = find_similarity_transform3d(x, y, w)
            return fit.rotation, fit.translation, fit.scale

        self.gradcheck(aligned, (src, dst, weights), requires_grad=(True, True, True))

    @pytest.mark.applies_to(devices=("cpu",), reason="CPU float64 second derivative reference")
    def test_isotropic_double_backward(self):
        eye = torch.eye(3, dtype=torch.float64, device="cpu")
        src = torch.cat((eye, -eye))[None].requires_grad_()
        dst = 2 * src.detach() + src.new_tensor([1, -2, 3])

        def aligned(points):
            return find_similarity_transform3d(points, dst)[:3]

        assert torch.autograd.gradgradcheck(aligned, (src,))

    @pytest.mark.applies_to(devices=("cpu",), reason="CPU bfloat16 autocast")
    def test_autocast_matches_float32(self):
        src = torch.tensor([[[0, 0, 0], [0, 1, 0], [2, 1, -1], [3, -1, 2]]], dtype=torch.float32, device="cpu")
        dst = src * 1.77 + src.new_tensor([0.81, -1.285, 2.75])
        expected = find_similarity_transform3d(src, dst)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = find_similarity_transform3d(src, dst)
        for output, reference in zip(actual, expected):
            self.assert_close(output, reference, rtol=0, atol=0)

    def test_dynamo(self, device, dtype, torch_optimizer):
        points = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], device=device, dtype=dtype)
        expected = find_similarity_transform3d(points, 2 * points + 1)
        actual = torch_optimizer(find_similarity_transform3d, fullgraph=True)(points, 2 * points + 1)
        for output, reference in zip(actual, expected):
            self.assert_close(output, reference)

    def test_promoted_dtype(self, device):
        points = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], device=device, dtype=torch.float16)
        fit = find_similarity_transform3d(points, points.float() + 2)
        assert fit.rotation.dtype == fit.translation.dtype == fit.scale.dtype == torch.float32
        self.assert_close(fit.translation, points.new_full((1, 3), 2, dtype=torch.float32))

    def test_incompatible_shapes(self, device, dtype):
        points = torch.zeros(1, 3, 3, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="at least three correspondences"):
            find_similarity_transform3d(points[:, :2], points[:, :2])
        with pytest.raises(ValueError, match="batch and correspondence dimensions"):
            find_similarity_transform3d(points, points, points.new_ones((1, 2)))
