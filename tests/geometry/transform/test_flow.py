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

from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip
from kornia.geometry.transform import warp_optical_flow

from testing.base import BaseTester


class TestWarpOpticalFlow(BaseTester):
    def test_convention_flip_replays_the_geometry_of_an_image_pair(self, device, dtype):
        augmenter = AugmentationSequential(
            RandomHorizontalFlip(p=1.0), data_keys=["input", "input"], transformation_matrix_mode="rigid"
        ).to(device=device, dtype=dtype)
        image0 = torch.arange(45, device=device, dtype=dtype).reshape(1, 1, 5, 9) / 45
        image1 = image0.flip(-2)
        flow = torch.cat((image0, image1), dim=1)
        known = image0 < 0.95
        aug0, aug1 = augmenter(image0, image1)
        matrix = augmenter.transform_matrix
        assert matrix is not None
        out, valid = warp_optical_flow(flow, matrix, aug0.shape[-2:], known)
        expected = flow.flip(-1) * flow.new_tensor([-1, 1]).view(1, 2, 1, 1)
        self.assert_close(aug0, image0.flip(-1))
        self.assert_close(aug1, image1.flip(-1))
        self.assert_close(valid, known.flip(-1))
        self.assert_close(out, torch.where(valid, expected, 0))

    def test_convention_projective_warp_transforms_correspondence_endpoints(self, device, dtype):
        x = torch.arange(5, device=device, dtype=dtype).reshape(1, 1, 1, 5).expand(1, 1, 4, 5)
        y = torch.arange(4, device=device, dtype=dtype).reshape(1, 1, 4, 1).expand(1, 1, 4, 5)
        flow = torch.cat((0.5 + 0.25 * x, -0.25 + 0.125 * y), dim=1)
        H = flow.new_tensor([[[1, 0.1, 0.4], [-0.2, 1, 0.3], [0.05, -0.02, 1]]])
        out, valid = warp_optical_flow(flow, H, (3, 4))
        # NumPy reference: p = dehom(np.linalg.inv(H) @ [x, y, 1]); s = np.clip(p, 0, [4, 3]).
        # The linear source field lets us evaluate bilinear interpolation at s without a sampling kernel.
        # For p inside the source footprint, expected = dehom(H @ [*(p + [0.5+.25*s[0], -.25+.125*s[1]]), 1]) - [x, y].
        expected = flow.new_tensor(
            [
                [
                    [
                        [0.465999808, 0.558632093, 0.714476895, 0.849786633],
                        [0.484984840, 0.557840044, 0.719602766, 0.858112724],
                        [0.0, 0.559643277, 0.724758381, 0.866486643],
                    ],
                    [
                        [-0.343368279, -0.356217802, -0.378922579, -0.381999958],
                        [-0.308580779, -0.295800964, -0.309867304, -0.323652105],
                        [0.0, -0.209488436, -0.234608653, -0.259224643],
                    ],
                ]
            ]
        )
        mask = torch.ones((1, 1, 3, 4), device=device, dtype=torch.bool)
        mask[0, 0, 2, 0] = False
        self.assert_close(valid, mask)
        self.assert_close(out, expected)
        scaled, scaled_valid = warp_optical_flow(flow, -H * 2, (3, 4))
        self.assert_close(scaled, out)
        self.assert_close(scaled_valid, valid)

    def test_convention_invalid_neighbors_have_no_weight_in_integer_samples(self, device, dtype):
        flow = torch.tensor(
            [[[[1, float("nan"), 4], [2, 3, 4]], [[2, float("nan"), 1], [1, 2, 3]]]], device=device, dtype=dtype
        )
        known = torch.ones_like(flow[:, :1], dtype=torch.bool)
        known[0, 0, 1, 2] = False
        H = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(2, 1, 1)
        H[1, 0, 2] = -0.5
        result, valid = warp_optical_flow(flow.expand(2, -1, -1, -1), H, (2, 2), known.expand(2, -1, -1, -1))
        expected_valid = torch.tensor(
            [[[[True, False], [True, True]]], [[[False, False], [True, False]]]], device=device
        )
        expected_result = flow.new_tensor(
            [[[[1, 0], [2, 3]], [[2, 0], [1, 2]]], [[[0, 0], [2.5, 0]], [[0, 0], [1.5, 0]]]]
        )
        self.assert_close(valid, expected_valid)
        self.assert_close(result, expected_result)

    def test_projects_outside_endpoint_without_invalidating_its_label(self, device, dtype):
        flow = torch.tensor([100.0, -25.0], device=device, dtype=dtype).reshape(1, 2, 1, 1)
        # A half-pixel back-projection on the footprint of a singleton row and column.
        H = flow.new_tensor([[[2, 0, 1], [0, 2, 1], [0, 0, 1]]])
        result, valid = warp_optical_flow(flow, H, (1, 1))
        self.assert_close(result, 2 * flow)
        assert bool(valid.all())

    def test_singleton_flow_preserves_footprint_and_accumulates_gradients(self, device, dtype):
        flow = torch.tensor([[5, -13], [-8, 9]], device=device, dtype=dtype).reshape(2, 2, 1, 1).requires_grad_()
        known = torch.tensor([True, False], device=device).reshape(2, 1, 1, 1)
        H = flow.new_tensor([[[2, 0, 1], [0, 2, 1], [0, 0, 1]]]).expand(2, -1, -1)
        result, valid = warp_optical_flow(flow, H, (3, 3), known)
        # The source domain [-0.5, 0.5) maps onto [0, 2), so only
        # the four integer centres in the top-left quadrant are in bounds.
        expected_valid = torch.zeros_like(valid)
        expected_valid[0, :, :2, :2] = True
        expected = torch.zeros_like(result)
        expected[0, :, :2, :2] = flow.new_tensor([10, -26]).view(2, 1, 1)
        self.assert_close(valid, expected_valid)
        self.assert_close(result, expected)
        (gradient,) = torch.autograd.grad(result.sum(), flow)
        self.assert_close(gradient, flow.new_tensor([8, 8, 0, 0]).reshape_as(flow))

    def test_projective_horizons_and_singular_batch_have_finite_backward(self, device, dtype):
        flow = torch.zeros((3, 2, 2, 4), device=device, dtype=dtype)
        flow[0, 0, :, 0] = 2  # p + u = (2, y) maps to the horizon under H[0].
        H = flow.new_tensor(
            [
                [[0.5, 0, 0], [0, 1, 0], [-0.5, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0.5, 0, 1]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
            ]
        )
        flow.requires_grad_()
        H.requires_grad_()
        result, valid = warp_optical_flow(flow, H, (1, 3))
        self.assert_close(
            valid, torch.tensor([[[[False, True, True]]], [[[True, True, False]]], [[[False] * 3]]], device=device)
        )
        self.assert_close(result, torch.zeros_like(result))
        for derivative in torch.autograd.grad(result.sum(), (flow, H)):
            assert bool(derivative.isfinite().all())

    @pytest.mark.applies_to(dtypes=("float16", "bfloat16"), reason="preserve pixel coordinates in half precision")
    def test_float32_homography_keeps_large_flip_coordinates_exact(self, device, dtype):
        width = 2050
        x = (torch.arange(width, device=device) % 5).to(dtype=dtype).reshape(1, 1, 1, width)
        flow = torch.cat((x, 1 + x), dim=1)
        H = torch.tensor([[[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]]], device=device, dtype=torch.float32)
        result, valid = warp_optical_flow(flow, H, (1, width))
        self.assert_close(result, flow.flip(-1) * flow.new_tensor([-1, 1]).reshape(1, 2, 1, 1))
        assert bool(valid.all())

    def test_gradcheck(self, device):
        generator = torch.Generator().manual_seed(124)
        flow = 0.125 * torch.rand((1, 2, 4, 5), generator=generator, dtype=torch.float64).to(device)
        H = flow.new_tensor([[[1.1, 0.06, -0.51], [-0.11, 1, -0.34], [0.04, 0.01, 1]]])
        self.gradcheck(lambda u, h: warp_optical_flow(u, h, (2, 3))[0], (flow, H))

    @pytest.mark.applies_to(dtypes=("float32", "float64"), reason="compare graph execution in the working dtype")
    def test_dynamo(self, device, dtype, torch_optimizer):
        flow = torch.ones((2, 2, 4, 5), device=device, dtype=dtype)
        H = flow.new_tensor([[[1, 0, -0.15], [0, 1, -0.25], [0.01, 0.03, 1]]]).expand(2, -1, -1)
        fn = torch_optimizer(warp_optical_flow)
        actual, valid = fn(flow, H, (3, 4))
        expected, expected_valid = warp_optical_flow(flow, H, (3, 4))
        self.assert_close(actual, expected)
        self.assert_close(valid, expected_valid)
