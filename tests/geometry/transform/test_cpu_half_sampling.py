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
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from kornia.augmentation import RandomAffine
from kornia.core.utils import _torch_grid_sample
from kornia.geometry import DepthWarper, PinholeCamera, warp_frame_depth
from kornia.geometry.grid import create_meshgrid
from kornia.geometry.transform import (
    HomographyWarper,
    elastic_transform2d,
    homography_warp,
    homography_warp3d,
    remap,
    rotate,
    warp_affine,
    warp_affine3d,
    warp_image_tps,
    warp_perspective,
    warp_perspective3d,
)
from kornia.geometry.transform.imgwarp import _runs_in_float32

from testing.base import BaseTester


class TestCPUHalfSampling(BaseTester):
    @staticmethod
    def _warp(image, operation, align_corners):
        batch, _, height, width = image.shape
        matrix = torch.eye(3, device=image.device, dtype=image.dtype)[None].expand(batch, -1, -1)
        options = {"align_corners": align_corners}
        if operation == "identity":
            return warp_affine(image, matrix[:, :2], (height, width), **options)
        if operation == "rotate":
            return rotate(image, image.new_tensor([25.0]), **options)
        if operation == "fill":
            matrix = matrix.clone()
            matrix[:, 0, 2] = 13.5
            return warp_affine(
                image,
                matrix[:, :2],
                (height, width),
                padding_mode="fill",
                fill_value=image.new_tensor([0.25, 0.5, 0.75]),
                **options,
            )
        if operation == "perspective":
            return warp_perspective(image, matrix, (height, width), **options)
        if operation == "remap":
            grid = create_meshgrid(height, width, device=image.device, dtype=image.dtype)
            return remap(
                image,
                grid[..., 0],
                grid[..., 1],
                normalized_coordinates=True,
                mode="nearest",
                padding_mode="reflection",
                **options,
            )
        if operation == "homography":
            return homography_warp(image, matrix, (height, width), padding_mode="border", **options)
        if operation == "cached_homography":
            warper = HomographyWarper(height, width, **options)
            warper.precompute_warp_grid(matrix)
            return warper(image)
        if operation == "elastic":
            noise = image.new_zeros(batch, 2, height, width)
            return elastic_transform2d(image, noise, kernel_size=(3, 3), sigma=(1.0, 1.0), **options)
        if operation == "tps":
            centers = image.new_tensor([[[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]]])
            weights = torch.zeros_like(centers)
            affine = image.new_tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
            return warp_image_tps(image, centers, weights, affine, **options)
        if operation == "depth":
            return warp_frame_depth(
                image, torch.ones_like(image[:, :1]), torch.eye(4, device=image.device, dtype=image.dtype)[None], matrix
            )
        if operation == "depth_warper":
            identity = torch.eye(4, device=image.device, dtype=image.dtype)[None]
            camera = PinholeCamera(identity, identity.clone(), image.new_tensor([height]), image.new_tensor([width]))
            warper = DepthWarper(camera, height, width, **options)
            warper.compute_projection_matrix(camera)
            return warper(torch.ones_like(image[:, :1]), image)
        if operation == "augmentation":
            aug = RandomAffine(degrees=(25.0, 25.0), p=1.0, **options)
            return aug(image)
        raise AssertionError(operation)

    # These operations hand grid_sample a grid they built in the image dtype; only the sampling moves to float32.
    @pytest.mark.parametrize(
        "operation", ["homography", "cached_homography", "elastic", "tps", "depth", "depth_warper"]
    )
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_float32_sampling_reference(self, device, dtype, operation, align_corners):
        if device.type != "cpu":
            pytest.skip("CPU sampling regression")
        image = ((torch.arange(3 * 256 * 256, device=device) % 251).float() / 251).reshape(1, 3, 256, 256).to(dtype)
        sample = F.grid_sample

        def reference(src, grid, **kwargs):
            work_dtype = torch.float64 if src.dtype == torch.float64 else torch.float32
            return sample(src.to(work_dtype), grid.to(work_dtype), **kwargs).to(src.dtype)

        with patch.object(F, "grid_sample", side_effect=reference):
            expected = self._warp(image, operation, align_corners)
        actual = self._warp(image, operation, align_corners)
        assert actual.dtype == dtype
        assert torch.isfinite(actual).all()
        self.assert_close(actual, expected, atol=0, rtol=0)

    @staticmethod
    def _float32_warp(image, operation, align_corners, param_dtype):
        # The transform is rounded to ``param_dtype`` first, so a float64 run of the same inputs is the reference.
        batch, _, height, width = image.shape
        c = math.cos(math.radians(25.0))
        s = math.sin(math.radians(25.0))
        cx, cy = (width - 1) / 2, (height - 1) / 2
        rotation = (
            torch.tensor([[[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy]]], dtype=torch.float64)
            .to(param_dtype)
            .to(image)
        )
        homography = torch.tensor([[[0.9, 0.05, 7.0], [-0.04, 1.1, -3.0], [1e-4, -2e-4, 1.0]]], dtype=torch.float64)
        homography = homography.to(param_dtype).to(image)
        options = {"align_corners": align_corners}
        if operation == "affine":
            return warp_affine(image, rotation, (height, width), **options)
        if operation == "affine_fill":
            fill = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64).to(param_dtype).to(image)
            return warp_affine(image, rotation, (height, width), padding_mode="fill", fill_value=fill, **options)
        if operation == "perspective":
            return warp_perspective(image, homography, (height, width), **options)
        if operation == "homography_pixel":
            return homography_warp(image, homography, (height, width), normalized_homography=False, **options)
        if operation == "remap_pixel":
            grid = create_meshgrid(height, width, False, dtype=torch.float64)
            points = torch.cat([grid, torch.ones_like(grid[..., :1])], -1) @ rotation.double()[0].T
            points = points.to(param_dtype).to(image)
            return remap(image, points[..., 0], points[..., 1], **options)
        if operation == "affine3d":
            volume = image[:, :1].reshape(batch, 1, 64, 32, 32)
            matrix = torch.tensor([[[c, -s, 0.0, 9.0], [s, c, 0.0, -6.0], [0.0, 0.0, 1.0, 1.5]]], dtype=torch.float64)
            return warp_affine3d(volume, matrix.to(param_dtype).to(image), (64, 32, 32), align_corners=align_corners)
        if operation == "perspective3d":
            volume = image[:, :1].reshape(batch, 1, 64, 32, 32)
            matrix = torch.eye(4, dtype=torch.float64)[None]
            matrix[:, :3, :3] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
            matrix[:, :3, 3] = torch.tensor([9.0, -6.0, 1.5], dtype=torch.float64)
            return warp_perspective3d(
                volume, matrix.to(param_dtype).to(image), (64, 32, 32), align_corners=align_corners
            )
        if operation == "homography3d":
            volume = image[:, :1].reshape(batch, 1, 64, 32, 32)
            matrix = torch.eye(4, dtype=torch.float64)[None]
            matrix[:, :3, :3] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
            matrix[:, :3, 3] = torch.tensor([0.1, -0.05, 0.02], dtype=torch.float64)
            return homography_warp3d(
                volume, matrix.to(param_dtype).to(image), (64, 32, 32), align_corners=align_corners
            )
        raise AssertionError(operation)

    @pytest.mark.parametrize(
        "operation",
        [
            "affine",
            "affine_fill",
            "perspective",
            "homography_pixel",
            "remap_pixel",
            "affine3d",
            "perspective3d",
            "homography3d",
        ],
    )
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_cpu_half_warp_matches_float64(self, device, dtype, operation, align_corners):
        # torch's CPU half grid_sample is corrupt from about 128 px, and a half grid is off by more than half a
        # pixel at 256 px in bfloat16 (#5751): the matrix warps and remap run in float32, so they agree with a
        # float64 run of the same half inputs to the output rounding.
        if device.type != "cpu" or dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("CPU float16/bfloat16 warps run in float32")
        image = ((torch.arange(3 * 256 * 256, device=device) % 251).float() / 251).reshape(1, 3, 256, 256).to(dtype)
        actual = self._float32_warp(image, operation, align_corners, dtype)
        expected = self._float32_warp(image.double(), operation, align_corners, dtype)
        assert actual.dtype == dtype
        self.assert_close(actual.double(), expected, rtol=0, atol=torch.finfo(dtype).eps)

    def test_cpu_half_augmentation_matches_float64(self, device, dtype):
        if device.type != "cpu" or dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("CPU float16/bfloat16 warps run in float32")
        image = ((torch.arange(3 * 256 * 256, device=device) % 251).float() / 251).reshape(1, 3, 256, 256).to(dtype)
        aug = RandomAffine(degrees=(25.0, 25.0), p=1.0)
        actual = aug(image)
        matrix = aug.transform_matrix[:, :2].double()
        expected = warp_affine(image.double(), matrix, (256, 256), align_corners=False)
        self.assert_close(actual.double(), expected, rtol=0, atol=torch.finfo(dtype).eps)

    @pytest.mark.parametrize("operation", ["affine", "perspective", "remap_pixel"])
    def test_cpu_half_warp_gradients_run_in_float32(self, device, dtype, operation):
        # Forward and backward both run in float32: the half gradients are the float32 ones, rounded.
        if device.type != "cpu" or dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("CPU float16/bfloat16 warps run in float32")
        image = torch.linspace(0, 1, 3 * 160 * 144, device=device).reshape(1, 3, 160, 144).to(dtype)
        half_image = image.clone().requires_grad_()
        float_image = image.float().requires_grad_()
        actual = self._float32_warp(half_image, operation, True, dtype)
        expected = self._float32_warp(float_image, operation, True, dtype).to(dtype)
        self.assert_close(actual, expected, rtol=0, atol=0)
        (actual_grad,) = torch.autograd.grad(actual.float().square().sum(), half_image)
        (expected_grad,) = torch.autograd.grad(expected.float().square().sum(), float_image)
        assert actual_grad.dtype == dtype
        self.assert_close(actual_grad, expected_grad.to(dtype), rtol=0, atol=0)

    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_sampling_gradients(self, device, dtype, mode, padding_mode, align_corners):
        if device.type != "cpu":
            pytest.skip("CPU sampling regression")
        image = torch.linspace(0, 1, 17 * 19, device=device, dtype=dtype).reshape(1, 1, 17, 19)
        image.requires_grad_()
        grid = create_meshgrid(13, 11, device=device, dtype=dtype).mul(1.2).add(0.03).requires_grad_()
        options = {"mode": mode, "padding_mode": padding_mode, "align_corners": align_corners}
        actual = remap(image, grid[..., 0], grid[..., 1], normalized_coordinates=True, **options)
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        expected = F.grid_sample(image.to(work_dtype), grid.to(work_dtype).contiguous(), **options).to(dtype)
        actual_grads = torch.autograd.grad(actual.sum(), (image, grid))
        expected_grads = torch.autograd.grad(expected.sum(), (image, grid))
        self.assert_close(actual, expected, atol=0, rtol=0)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            assert actual_grad.dtype == dtype
            assert torch.isfinite(actual_grad).all()
            self.assert_close(actual_grad, expected_grad, atol=0, rtol=0)

    @pytest.mark.parametrize("size", [0, 4])
    @pytest.mark.parametrize("map_dtype", [torch.float32, torch.int64])
    def test_mismatched_grid_dtype(self, device, dtype, size, map_dtype):
        if device.type != "cpu" or dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("CPU half sampling dtype validation")
        image = torch.ones(1, 1, 4, 4, device=device, dtype=dtype)
        coords = torch.zeros(1, size, 4, device=device, dtype=map_dtype)
        with pytest.raises(RuntimeError, match=r"(dtype|scalar type|grid_sampler_2d_cpu not implemented)"):
            remap(image, coords, coords, normalized_coordinates=True)

    def test_mismatched_matrix_dtype(self, device, dtype):
        # A float32 matrix next to a half image is rejected as before; only matching dtypes move to float32.
        if device.type != "cpu" or dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("CPU half sampling dtype validation")
        image = torch.ones(1, 1, 4, 4, device=device, dtype=dtype)
        matrix = torch.eye(3, device=device)[None]
        with pytest.raises(RuntimeError, match=r"(dtype|scalar type|grid_sampler_2d_cpu not implemented)"):
            warp_affine(image, matrix[:, :2], (4, 4))
        with pytest.raises(RuntimeError, match=r"(dtype|scalar type|grid_sampler_2d_cpu not implemented)"):
            warp_perspective(image, matrix, (4, 4))

    @pytest.mark.parametrize("half", [torch.float16, torch.bfloat16], ids=["float16", "bfloat16"])
    def test_non_cpu_half_keeps_its_own_dtype(self, half):
        # Only CPU half warps move to float32 (#5751); other devices keep their own half kernels. The meta
        # device stands in for them, so every CI leg checks the device gate.
        image = torch.zeros(1, 1, 4, 4, device="meta", dtype=half)
        grid = torch.zeros(1, 4, 4, 2, device="meta", dtype=half)
        assert not _runs_in_float32(image, image[:, 0])
        seen = []

        def record(src, *args, **kwargs):
            seen.append(src.dtype)
            return src

        with patch.object(F, "grid_sample", side_effect=record):
            _torch_grid_sample(image, grid)
        assert seen == [half]
