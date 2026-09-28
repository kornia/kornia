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
import sys
import warnings

import pytest
import torch

import kornia
from kornia.core._compat import torch_version
from kornia.geometry import transform_points
from kornia.geometry.conversions import denormalize_homography
from kornia.geometry.transform import Homography, ImageRegistrator, Similarity, homography_warp

from testing.base import BaseTester, supports_bilinear_2d_grid_sample_backward, supports_reflect_padding
from testing.casts import dict_to


class TestSimilarity(BaseTester):
    def test_smoke(self, device, dtype):
        expected = torch.eye(3, device=device, dtype=dtype)[None]
        for r, sc, sh in zip([True, False], [True, False], [True, False]):
            sim = kornia.geometry.transform.Similarity(r, sc, sh).to(device, dtype)
            self.assert_close(sim(), expected, atol=1e-4, rtol=1e-4)

    def test_smoke_inverse(self, device, dtype):
        expected = torch.eye(3, device=device, dtype=dtype)[None]
        for r, sc, sh in zip([True, False], [True, False], [True, False]):
            s = kornia.geometry.transform.Similarity(r, sc, sh).to(device, dtype)
            self.assert_close(s.forward_inverse(), expected, atol=1e-4, rtol=1e-4)

    def test_scale(self, device, dtype):
        sc = 0.5
        sim = kornia.geometry.transform.Similarity(True, True, True).to(device, dtype)
        sim.scale.data *= sc
        expected = torch.tensor([[0.5, 0, 0.0], [0, 0.5, 0], [0, 0, 1]], device=device, dtype=dtype)[None]
        inv_expected = torch.tensor([[2.0, 0, 0.0], [0, 2.0, 0], [0, 0, 1]], device=device, dtype=dtype)[None]
        self.assert_close(sim.forward_inverse(), inv_expected, atol=1e-4, rtol=1e-4)
        self.assert_close(sim(), expected, atol=1e-4, rtol=1e-4)

    def test_repr(self, device, dtype):
        for r, sc, sh in zip([True, False], [True, False], [True, False]):
            s = kornia.geometry.transform.Similarity(r, sc, sh).to(device, dtype)
            assert s is not None


class TestHomography(BaseTester):
    def test_smoke(self, device, dtype):
        expected = torch.eye(3, device=device, dtype=dtype)[None]
        h = kornia.geometry.transform.Homography().to(device, dtype)
        self.assert_close(h(), expected, atol=1e-4, rtol=1e-4)

    def test_smoke_inverse(self, device, dtype):
        expected = torch.eye(3, device=device, dtype=dtype)[None]
        h = kornia.geometry.transform.Homography().to(device, dtype)
        self.assert_close(h.forward_inverse(), expected, atol=1e-4, rtol=1e-4)

    def test_repr(self, device, dtype):
        h = kornia.geometry.transform.Homography().to(device, dtype)
        assert h is not None


class TestImageRegistrator(BaseTester):
    @pytest.mark.parametrize("model_type", ["homography", "similarity", "translation", "scale", "rotation"])
    def test_smoke(self, device, dtype, model_type):
        ir = kornia.geometry.transform.ImageRegistrator(model_type).to(device, dtype)
        assert ir is not None

    @pytest.mark.slow
    @pytest.mark.xfail(
        torch_version() in {"2.0.0", "2.0.1", "2.1.2", "2.2.2", "2.4.0"}, reason="failing at some 2.x torch"
    )
    def test_registration_toy(self, device, dtype):
        ch, height, width = 3, 16, 18
        homography = torch.eye(3, device=device, dtype=dtype)[None]
        homography[..., 0, 0] = 1.05
        homography[..., 1, 1] = 1.05
        homography[..., 0, 2] = 0.01
        img_src = torch.rand(1, ch, height, width, device=device, dtype=dtype)
        img_dst = kornia.geometry.homography_warp(img_src, homography, (height, width), align_corners=False)
        IR = ImageRegistrator("Similarity", num_iterations=500, lr=3e-4, pyramid_levels=2).to(device, dtype)
        model = IR.register(img_src, img_dst)
        self.assert_close(model, homography, atol=1e-3, rtol=1e-3)
        model, intermediate = IR.register(img_src, img_dst, output_intermediate_models=True)
        assert len(intermediate) == 2

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    @pytest.mark.skipif(
        torch_version() == "2.0.0" and "win" in sys.platform, reason="Tensor not matching on win with torch 2.0"
    )
    def test_registration_real(self, device, dtype, data):
        data_dev = dict_to(data, device, dtype)
        IR = ImageRegistrator("homography", num_iterations=1200, lr=2e-2, pyramid_levels=5).to(device, dtype)
        model = IR.register(data_dev["image0"], data_dev["image1"])
        homography_gt = torch.inverse(data_dev["H_gt"])
        homography_gt = homography_gt / homography_gt[2, 2]
        h0, w0 = data["image0"].shape[2], data["image0"].shape[3]
        h1, w1 = data["image1"].shape[2], data["image1"].shape[3]

        model_denormalized = denormalize_homography(model, (h0, w0), (h1, w1))
        model_denormalized = model_denormalized / model_denormalized[0, 2, 2]

        bbox = torch.tensor([[[0, 0], [w0, 0], [w0, h0], [0, h0]]], device=device, dtype=dtype)
        bbox_in_2_gt = transform_points(homography_gt[None], bbox)
        bbox_in_2_gt_est = transform_points(model_denormalized, bbox)
        # The tolerance is huge, because the error is in pixels
        # and transformation is quite significant, so
        # 15 px  reprojection error is not super huge
        self.assert_close(bbox_in_2_gt, bbox_in_2_gt_est, atol=15, rtol=0.1)

    def test_register_with_shape_mismatch(self, device):
        img1 = torch.rand(1, 1, 64, 64, device=device)
        img2 = torch.rand(1, 1, 32, 32, device=device)

        registrator = ImageRegistrator("similarity", allow_shape_mismatch=True, pyramid_levels=1, num_iterations=1).to(
            device
        )

        out = registrator.register(img1, img2)

        assert out is not None

    def test_register_shape_mismatch_raises(self, device):
        img1 = torch.rand(1, 1, 64, 64, device=device)
        img2 = torch.rand(1, 1, 32, 32, device=device)

        registrator = ImageRegistrator("similarity", allow_shape_mismatch=False)

        with pytest.raises(ValueError):
            registrator.register(img1, img2)

    def test_warp_dst_into_src_and_deprecated_alias(self, device, dtype):
        ch, height, width = 1, 8, 10
        ir = ImageRegistrator("Similarity").to(device, dtype)
        # A pure shift, so that the inverse warp differs from the forward one and equals the forward warp of the
        # opposite shift. An identity model cannot tell the two directions apart.
        with torch.no_grad():
            ir.model.shift.copy_(torch.tensor([[[0.5], [-0.25]]], device=device, dtype=dtype))
        opposite = ImageRegistrator("Similarity").to(device, dtype)
        with torch.no_grad():
            opposite.model.shift.copy_(-ir.model.shift)
        dst = torch.rand(1, ch, height, width, device=device, dtype=dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            into_src = ir.warp_dst_into_src(dst)
        self.assert_close(into_src, opposite.warp_src_into_dst(dst))
        assert not torch.allclose(into_src, ir.warp_src_into_dst(dst))

        with pytest.warns(DeprecationWarning, match="`warp_dst_inro_src` is deprecated in favor of"):
            via_alias = ir.warp_dst_inro_src(dst)
        self.assert_close(via_alias, into_src)


class TestConventionsImageRegistrator(BaseTester):
    height, width = 32, 48

    def _scene(self, dx: float, dy: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Four Gaussian blobs, moved by (dx, dy) pixels. Analytic, so the fixture needs no seed; register() draws no
        # random numbers either, so every run is deterministic.
        ys, xs = torch.meshgrid(
            torch.arange(self.height, device=device, dtype=dtype),
            torch.arange(self.width, device=device, dtype=dtype),
            indexing="ij",
        )
        blobs = [(12.0, 9.0, 4.0, 1.0), (30.0, 20.0, 6.0, 0.8), (38.0, 7.0, 3.0, 0.6), (20.0, 24.0, 3.5, 0.9)]
        img = torch.zeros(self.height, self.width, device=device, dtype=dtype)
        for cx, cy, sigma, amplitude in blobs:
            img = img + amplitude * torch.exp(-((xs - dx - cx) ** 2 + (ys - dy - cy) ** 2) / (2 * sigma**2))
        return img[None, None]

    @staticmethod
    def _skip_without_kernels(device: torch.device, dtype: torch.dtype) -> None:
        # Registration differentiates through a bilinear grid_sample and builds its pyramid with reflection padding.
        # torch 2.5.1 has neither kernel for float16 on CPU, nor the grid_sample one for bfloat16.
        if not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip(f"torch has no {dtype} bilinear grid_sample kernel with backward on {device.type}")
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"torch has no {dtype} reflection padding kernel on {device.type}")

    def _register(
        self, device: torch.device, dtype: torch.dtype, num_iterations: int, pyramid_levels: int
    ) -> tuple[ImageRegistrator, torch.Tensor, torch.Tensor, torch.Tensor]:
        src = self._scene(0.0, 0.0, device, dtype)
        dst = self._scene(3.0, -1.5, device, dtype)  # the content moves by (+3, -1.5) px from src to dst
        ir = ImageRegistrator(
            "translation", lr=1e-2, num_iterations=num_iterations, pyramid_levels=pyramid_levels, tolerance=1e-9
        ).to(device, dtype)
        model = ir.register(src, dst).detach()
        return ir, model, src, dst

    def test_convention_image_registrator_model_maps_dst_to_src(self, device, dtype):
        # register(src, dst) returns the (1, 3, 3) model that maps destination coordinates to source coordinates
        # (homography_warp's src_homo_dst: a pull warp), in normalised [-1, 1] coordinates with align_corners=False.
        # Content moved by (+3, -1.5) px from src to dst gives a model shift of (-3, +1.5) px; the inverse model would
        # give (+3, -1.5).
        self._skip_without_kernels(device, dtype)
        _, model, _, _ = self._register(device, dtype, num_iterations=60, pyramid_levels=2)
        size = (self.height, self.width)
        shift = denormalize_homography(model.float(), size, size, align_corners=False)[0, :2, 2]
        # Measured about (-3.0, 1.5); the 0.5 px margins absorb the half-precision error.
        assert -3.5 < shift[0].item() < -2.5
        assert 1.0 < shift[1].item() < 2.0

    def test_convention_image_registrator_warp_matches_homography_warp(self, device, dtype):
        # warp_src_into_dst(src) is homography_warp(src, model, (H, W), align_corners=False): the default
        # HomographyWarper reads the model in normalised coordinates with align_corners=False.
        self._skip_without_kernels(device, dtype)
        ir, model, src, _ = self._register(device, dtype, num_iterations=20, pyramid_levels=1)
        assert (model[0, :2, 2].abs() > 0.05).all()  # a model far from the identity
        size = (self.height, self.width)
        warped = ir.warp_src_into_dst(src).detach()
        self.assert_close(warped, homography_warp(src, model, size, align_corners=False))
        # The other align_corners convention samples elsewhere under this model (0.015 in float32).
        assert (warped - homography_warp(src, model, size, align_corners=True)).abs().max().item() > 5e-3

    def test_convention_image_registrator_register_resets_model(self, device, dtype):
        # A loaded state_dict drives the warps, but register() always starts again from the identity: the loaded
        # state is not a warm start, so a register() without iterations returns the identity.
        self._skip_without_kernels(device, dtype)
        ir, model, src, dst = self._register(device, dtype, num_iterations=20, pyramid_levels=1)
        assert (model[0, :2, 2].abs() > 0.05).all()  # a model far from the identity
        loaded = ImageRegistrator("translation", num_iterations=0).to(device, dtype)
        loaded.load_state_dict(ir.state_dict())
        self.assert_close(loaded.model().detach(), model)
        self.assert_close(loaded.warp_src_into_dst(src).detach(), ir.warp_src_into_dst(src).detach())
        identity = torch.eye(3, device=device, dtype=dtype)[None]
        self.assert_close(loaded.register(src, dst).detach(), identity)

    def test_convention_similarity_rotation_in_degrees(self, device, dtype):
        # Similarity.forward() is [[scale * R(rot), shift], [0, 0, 1]], with R = [[cos, sin], [-sin, cos]] from
        # angle_to_rotation_matrix: rot is in degrees.
        sim = Similarity().to(device, dtype)
        with torch.no_grad():
            sim.rot.fill_(30.0)
            sim.scale.fill_(1.5)
            sim.shift.copy_(torch.tensor([[[0.2], [-0.1]]], device=device, dtype=dtype))
        c, s = 1.5 * math.cos(math.radians(30.0)), 1.5 * math.sin(math.radians(30.0))  # 1.299, 0.75
        expected = torch.tensor([[[c, s, 0.2], [-s, c, -0.1], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        self.assert_close(sim().detach(), expected)

    @pytest.mark.parametrize(
        ("model_type", "optimized"),
        [
            ("homography", ["model"]),
            ("similarity", ["rot", "scale", "shift"]),
            ("translation", ["shift"]),
            ("rotation", ["rot"]),
            ("scale", ["scale"]),
        ],
    )
    def test_convention_image_registrator_model_type_parameters(self, model_type, optimized):
        # 'homography' optimizes a Homography; the other strings a Similarity that optimizes only the named parameters.
        ir = ImageRegistrator(model_type)
        assert isinstance(ir.model, Homography if model_type == "homography" else Similarity)
        assert sorted(name for name, _ in ir.model.named_parameters()) == optimized

    def test_convention_image_registrator_shape_mismatch_resizes_src(self, device, dtype):
        # allow_shape_mismatch=True resizes src_img to the height and width of dst_img before the loss, and a
        # different channel count still raises.
        self._skip_without_kernels(device, dtype)
        seen = []

        def l1(a: torch.Tensor, b: torch.Tensor, reduction: str = "none") -> torch.Tensor:
            seen.append((tuple(a.shape), tuple(b.shape)))
            return torch.nn.functional.l1_loss(a, b, reduction=reduction)

        ir = ImageRegistrator(
            "translation", num_iterations=1, pyramid_levels=1, loss_fn=l1, allow_shape_mismatch=True
        ).to(device, dtype)
        dst = self._scene(0.0, 0.0, device, dtype)
        ir.register(dst[..., ::2, ::2], dst)
        assert set(seen) == {((1, 1, self.height, self.width), (1, 1, self.height, self.width))}
        with pytest.raises(ValueError):
            ir.register(dst.expand(1, 3, -1, -1), dst)
