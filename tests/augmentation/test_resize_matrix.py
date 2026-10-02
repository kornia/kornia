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

from __future__ import annotations

import pytest
import torch

import kornia.augmentation as K
from kornia.constants import Resample
from kornia.geometry.transform import get_perspective_transform

from testing.base import BaseTester


class TestResizeMatrix(BaseTester):
    @staticmethod
    def _ramp(height, width, device, dtype):
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((x, y)).unsqueeze(0).repeat(2, 1, 1, 1)

    @staticmethod
    def _augmentation(kind, size, align_corners, device, dtype):
        if kind == "resize":
            aug = K.Resize(size, align_corners=align_corners)
        elif kind == "longest":
            aug = K.LongestMaxSize(size[1], align_corners=align_corners)
        elif kind == "smallest":
            aug = K.SmallestMaxSize(size[0], align_corners=align_corners)
        else:
            aug = K.RandomResizedCrop(size, align_corners=align_corners, cropping_mode="slice")
        # The True path solves in the parameter dtype; use double for a tight double control.
        aug.set_rng_device_and_dtype(device, torch.float64 if dtype == torch.float64 else torch.float32)
        return aug

    @pytest.mark.parametrize("kind", ["resize", "longest", "smallest", "crop"])
    @pytest.mark.parametrize("size", [(10, 14), (3, 4), (5, 7), (5, 11)])
    @pytest.mark.parametrize("align_corners", [False, True])
    def test_matrix_predicts_sampled_coordinates_4804(self, device, dtype, kind, size, align_corners):
        image = self._ramp(9, 12, device, dtype) if kind == "crop" else self._ramp(5, 7, device, dtype)
        aug = self._augmentation(kind, size, align_corners, device, dtype)
        params = aug.forward_parameters(image.shape)
        if kind == "crop":
            # Distinct origins and crop sizes expose translation signs and per-row scaling.
            params["src"] = image.new_tensor([[[1, 2], [7, 2], [7, 6], [1, 6]], [[3, 1], [10, 1], [10, 6], [3, 6]]])
        out = aug(image, params=params)
        assert aug._transform_matrix is None  # construction remains lazy
        matrix = aug.transform_matrix
        assert matrix.dtype == dtype
        assert matrix.device == image.device
        assert matrix.shape == (2, 3, 3)
        assert aug.transform_matrix is matrix
        src = params["src"].to(image)
        origin = src[:, 0]
        extent = src[:, 2] - origin
        h, w = out.shape[-2:]
        lengths = image.new_tensor([w, h])
        scales = (lengths - 1) / extent if align_corners else lengths / (extent + 1)
        offsets = -origin * scales
        if not align_corners:
            offsets = offsets + (scales - 1) / 2
        expected = torch.eye(3, device=device, dtype=dtype).repeat(2, 1, 1)
        expected[:, 0, 0], expected[:, 1, 1] = scales.unbind(-1)
        expected[:, :2, 2] = offsets
        tolerance = {"atol": 1e-12, "rtol": 1e-12} if dtype == torch.float64 else {}
        self.assert_close(matrix, expected, **tolerance)
        # Two ramps report the source x and y actually sampled. Invert the recorded
        # matrix, then account for interpolate's clamping to the crop boundary.
        grid = self._ramp(h, w, device, dtype).flatten(2)
        homogeneous = torch.cat((grid, torch.ones_like(grid[:, :1])), dim=1)
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        predicted = (torch.linalg.inv(matrix.to(work_dtype)) @ homogeneous.to(work_dtype))[:, :2]
        predicted = predicted.maximum(origin.unsqueeze(-1)).minimum((origin + extent).unsqueeze(-1))
        self.assert_close(out.flatten(2), predicted.to(dtype), **tolerance)

    @pytest.mark.parametrize("kind", ["resize", "crop"])
    @pytest.mark.parametrize("shape,size", [((1, 7), (3, 4)), ((5, 1), (2, 3)), ((5, 7), (1, 1)), ((1, 1), (1, 1))])
    def test_singleton_half_pixel_matrix_4804(self, device, dtype, kind, shape, size):
        image = self._ramp(*shape, device, dtype)
        aug = self._augmentation(kind, size, False, device, dtype)
        params = aug.forward_parameters(image.shape)
        h, w = shape
        params["src"] = image.new_tensor([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]]).repeat(2, 1, 1)
        out = aug(image, params=params)
        sx, sy = size[1] / w, size[0] / h
        expected = image.new_tensor([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        self.assert_close(aug.transform_matrix, expected.expand(2, -1, -1))
        self.assert_close(out, torch.nn.functional.interpolate(image, size, mode="bilinear", align_corners=False))

    @pytest.mark.parametrize("kind", ["resize", "longest", "smallest", "crop"])
    def test_sequential_annotations_follow_pixels_4804(self, device, dtype, kind):
        image = self._ramp(8, 12, device, dtype)
        aug = self._augmentation(kind, (4, 6), False, device, dtype)
        seq = K.AugmentationSequential(aug, data_keys=["input", "keypoints", "bbox"])
        params = seq.forward_parameters(image.shape)
        if kind == "crop":
            params[0].data["src"] = image.new_tensor(
                [[[2, 1], [9, 1], [9, 6], [2, 6]], [[1, 2], [8, 2], [8, 7], [1, 7]]]
            )
        # Select interior output pixels and derive their source centres analytically.
        src = params[0].data["src"].to(image)
        origin, extent = src[:, 0], src[:, 2] - src[:, 0] + 1
        target = image.new_tensor([[[1, 1], [4, 1], [4, 2], [1, 2]]]).repeat(2, 1, 1)
        points = (target + 0.5) * (extent / image.new_tensor([6, 4])).unsqueeze(1) - 0.5 + origin.unsqueeze(1)
        output, mapped, boxes = seq(image, points, points.unsqueeze(1), params=params)
        self.assert_close(mapped, target)
        self.assert_close(boxes, target.unsqueeze(1))
        self.assert_close(output[:, :, [1, 1, 2, 2], [1, 4, 4, 1]].transpose(1, 2), points)

    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("kind", ["resize", "crop"])
    def test_bicubic_matrix_and_image_4804(self, device, dtype, kind, align_corners):
        image = self._ramp(5, 7, device, dtype)
        aug = self._augmentation(kind, (8, 11), align_corners, device, dtype)
        params = aug.forward_parameters(image.shape)
        params["src"] = params["src"].new_tensor([[[0, 0], [6, 0], [6, 4], [0, 4]]]).repeat(2, 1, 1)
        aug(image, params=params)
        bilinear_matrix = aug.transform_matrix.clone()
        out = aug(image, params=params, resample=Resample.BICUBIC)
        self.assert_close(aug.transform_matrix, bilinear_matrix, atol=0, rtol=0)
        expected = torch.nn.functional.interpolate(image, (8, 11), mode="bicubic", align_corners=align_corners)
        self.assert_close(out, expected)

    @pytest.mark.parametrize("align_corners", [False, True])
    def test_resample_keeps_corner_geometry_4804(self, device, dtype, align_corners):
        image = self._ramp(8, 12, device, dtype)
        aug = K.RandomResizedCrop((4, 6), cropping_mode="resample", align_corners=align_corners)
        params = aug.forward_parameters(image.shape)
        output = aug(image, params=params)
        expected = get_perspective_transform(params["src"].to(image), params["dst"].to(image))
        self.assert_close(aug.transform_matrix, expected, atol=0, rtol=0)
        # Slice/resample agree for corner alignment; False intentionally uses distinct grids.
        if align_corners and dtype in (torch.float32, torch.float64):
            # Half-precision warp grids have additional rounding beyond interpolate.
            sliced = K.RandomResizedCrop((4, 6), align_corners=True)(image, params=params)
            self.assert_close(output, sliced)

    @pytest.mark.parametrize("kind", ["resize", "crop"])
    @pytest.mark.parametrize("resample,align_corners", [("bilinear", True), ("nearest", True), ("nearest", False)])
    def test_unchanged_matrix_paths(self, device, dtype, kind, resample, align_corners):
        image = self._ramp(5, 7, device, dtype)
        aug = self._augmentation(kind, (8, 11), align_corners, device, dtype)
        params = aug.forward_parameters(image.shape)
        if kind == "crop":
            # Random singleton crops make the unchanged corner-aligned solve singular.
            params["src"] = params["src"].new_tensor(
                [[[1, 1], [5, 1], [5, 4], [1, 4]], [[2, 0], [6, 0], [6, 3], [2, 3]]]
            )
        aug(image, params=params, resample=Resample.get(resample))
        src, dst = aug._params["src"], aug._params["dst"]
        if kind == "crop":
            src, dst = src.to(image), dst.to(image)
        expected = get_perspective_transform(src, dst).to(image)
        self.assert_close(aug.transform_matrix, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("kind", ["resize", "crop"])
    def test_gradcheck(self, device, kind):
        image = self._ramp(3, 4, device, torch.float64)[:1]
        points = image.new_tensor([[[1.25, 1.5]]])
        aug = self._augmentation(kind, (2, 3), False, device, torch.float64)
        seq = K.AugmentationSequential(aug, data_keys=["input", "keypoints"])
        params = seq.forward_parameters(image.shape)
        if kind == "crop":
            params[0].data["src"] = image.new_tensor([[[0, 0], [3, 0], [3, 2], [0, 2]]])
        self.gradcheck(lambda x, p: tuple(seq(x, p, params=params)), (image, points))

    @pytest.mark.parametrize("kind", ["resize", "longest", "smallest"])
    @pytest.mark.parametrize("align_corners", [False, True])
    def test_inverse_restores_ramp_4804(self, device, dtype, kind, align_corners):
        # Bilinear resampling reproduces a ramp, so warping a 2x upscale back with the recorded matrix restores
        # the interior exactly. The corner-to-corner matrix at align_corners=False missed by 0.21 px here.
        image = self._ramp(9, 13, device, dtype)
        aug = self._augmentation(kind, (18, 26), align_corners, device, dtype)
        restored = aug.inverse(aug(image))
        # Two half-precision warps round by up to one bfloat16 ulp at 12 (0.0625), below the 0.21 px defect.
        tolerances = {torch.float64: (1e-12, 1e-12), torch.float16: (0.05, 0), torch.bfloat16: (0.1, 0)}
        atol, rtol = tolerances.get(dtype, (1e-4, 1e-4))
        self.assert_close(restored[..., 1:-1, 1:-1], image[..., 1:-1, 1:-1], atol=atol, rtol=rtol)
