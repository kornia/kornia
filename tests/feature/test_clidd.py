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
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from kornia.core._compat import torch_version_lt
from kornia.core.exceptions import BaseError
from kornia.feature import CLIDD, CLIDDModel
from kornia.feature.clidd import clidd as clidd_module

from testing.base import BaseTester

_HAS_TRITON = importlib.util.find_spec("triton") is not None


def _bilinear(fmap: torch.Tensor, x: float, y: float) -> torch.Tensor:
    """Read ``fmap`` ``(C, h, w)`` bilinearly at map pixel ``(x, y)``, with zeros outside: written out by hand."""
    _, h, w = fmap.shape
    x0, y0 = int(torch.tensor(x).floor()), int(torch.tensor(y).floor())
    out = fmap.new_zeros(fmap.shape[0])
    for xi, wx in ((x0, x0 + 1 - x), (x0 + 1, x - x0)):
        for yi, wy in ((y0, y0 + 1 - y), (y0 + 1, y - y0)):
            if 0 <= xi < w and 0 <= yi < h:
                out = out + wx * wy * fmap[:, yi, xi]
    return out


def _image(batch: int = 1, height: int = 96, width: int = 128) -> torch.Tensor:
    """A deterministic textured RGB image in [0, 1]."""
    y, x = torch.meshgrid(torch.arange(float(height)), torch.arange(float(width)), indexing="ij")
    img = torch.stack(
        [0.5 + 0.5 * torch.sin(x / 3) * torch.cos(y / 5), 0.5 + 0.5 * torch.sin((x + y) / 7), (x * y % 17) / 16]
    )
    return img[None].repeat(batch, 1, 1, 1)


def _wide(device: torch.device) -> torch.dtype:
    """float64 for a reference computation, float32 on MPS, which has no float64."""
    return torch.float32 if device.type == "mps" else torch.float64


def _random_clidd(variant: str, device: torch.device, dtype: torch.dtype, **kwargs) -> CLIDD:
    torch.manual_seed(0)
    return CLIDD(variant, **kwargs).to(device, dtype).eval()


class TestCLIDDModel(BaseTester):
    @pytest.mark.parametrize("variant", ["A48", "S64", "E128"])
    def test_cardinality(self, variant, device, dtype):
        net = CLIDDModel(variant, use_triton=False).to(device, dtype).eval()
        cfg = clidd_module._CLIDD_CONFIGS[variant]
        maps, score = net(torch.rand(2, 3, 64, 96, device=device, dtype=dtype))
        assert maps[0].shape == (2, cfg["c1"], 32, 48)
        assert maps[1].shape == (2, cfg["c2"], 8, 12)
        assert maps[2].shape == (2, cfg["c3"], 2, 3)
        assert score.shape == (2, 1, 64, 96)
        desc = net.describe(maps, torch.rand(2, 7, 2, device=device, dtype=dtype) * 2 - 1)
        assert desc.shape == (2, 7, cfg["cdesc"])

    def test_state_dict_keys_match_the_reference_checkpoint(self):
        # Key names and shapes of the authors' A48.pth (v1.0 release), which from_pretrained loads with strict=True.
        state = CLIDDModel("A48").state_dict()
        assert len(state) == 62
        assert state["block1.0.layer.0.weight"].shape == (4, 3, 4, 4)
        assert state["block2.1.layer.3.weight"].shape == (4, 4, 3, 3)
        assert state["block3.1.layer.4.running_var"].shape == (4,)
        assert state["score_head.2.weight"].shape == (4, 4, 3, 3)
        assert state["dcn.weight"].shape == (24, 12, 1, 1)
        assert state["post_conv.weight"].shape == (48, 12, 1, 4)

    def test_unknown_variant(self):
        with pytest.raises(BaseError):
            CLIDDModel("X99")

    def test_convention_clidd_samples_each_map_at_the_keypoint_moved_by_its_offset(self, device):
        """A descriptor projects bilinear samples of each map at the keypoint plus the offsets of that map."""
        torch.manual_seed(0)
        wide = _wide(device)
        net = CLIDDModel("A48", use_triton=False).to(device, wide).eval()
        M = net.num_points
        # Zero offset weights: the bias alone gives the offsets, (x, y) pixels of the 64 x 96 image for map l, point m.
        offsets = torch.randn(3, M, 2, dtype=wide) * 3
        with torch.no_grad():
            net.dcn.weight.zero_()
            net.dcn.bias.copy_(offsets.flatten())
        maps = tuple(torch.randn(1, 4, 64 // s, 96 // s, device=device, dtype=wide) for s in (2, 8, 32))
        px, py = 37.0, 21.0  # pixel of the 64 x 96 image
        kp = torch.tensor([[[(2 * px + 1) / 96 - 1, (2 * py + 1) / 64 - 1]]], device=device, dtype=wide)
        desc = net.describe(maps, kp)[0, 0]

        samples = []
        for level, stride in enumerate((2, 8, 32)):
            cols = []
            for m in range(M):
                # Pixel centre p of the image lies at (p + 0.5) / stride - 0.5 of a map with that stride.
                x = (px + offsets[level, m, 0].item() + 0.5) / stride - 0.5
                y = (py + offsets[level, m, 1].item() + 0.5) / stride - 0.5
                cols.append(_bilinear(maps[level][0], x, y))
            samples.append(torch.stack(cols, -1))
        samples = torch.cat(samples, 0)  # (C_1 + C_2 + C_3, M)
        expected = torch.einsum("ocm,cm->o", net.post_conv.weight[:, :, 0], samples) + net.post_conv.bias
        self.assert_close(desc, F.normalize(expected, dim=0))

    def test_offsets_are_clamped_to_max_offset(self, device):
        wide = _wide(device)
        net = CLIDDModel("A48", max_offset=2.0, use_triton=False).to(device, wide).eval()
        far = CLIDDModel("A48", max_offset=2.0, use_triton=False).to(device, wide).eval()
        far.load_state_dict(net.state_dict())
        with torch.no_grad():
            net.dcn.weight.zero_()
            net.dcn.bias.fill_(2.0)
            far.dcn.weight.zero_()
            far.dcn.bias.fill_(50.0)
        maps = tuple(torch.randn(1, 4, 64 // s, 64 // s, device=device, dtype=wide) for s in (2, 8, 32))
        kp = torch.zeros(1, 3, 2, device=device, dtype=wide)
        self.assert_close(far.describe(maps, kp), net.describe(maps, kp))

    def test_gradcheck(self, device):
        net = CLIDDModel("A48", use_triton=False).to(device, torch.float64).eval()
        maps = [torch.rand(1, 4, 16 // s, 16 // s, device=device, dtype=torch.float64) for s in (1, 4, 16)]
        kp = torch.rand(1, 3, 2, device=device, dtype=torch.float64) * 1.6 - 0.8

        def fn(m0, m1, m2, k):
            return net.describe((m0, m1, m2), k)

        self.gradcheck(fn, (*maps, kp), requires_grad=[True, True, True, False])

    def test_dynamo(self, device, dtype, torch_optimizer):
        net = CLIDDModel("S64", use_triton=False).to(device, dtype).eval()
        x = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        kp = torch.rand(1, 9, 2, device=device, dtype=dtype) * 2 - 1

        def fn(img, k):
            maps, score = net(img)
            return net.describe(maps, k), score

        desc_c, score_c = torch_optimizer(fn)(x, kp)
        desc, score = fn(x, kp)
        self.assert_close(score_c, score, low_tolerance=True)
        self.assert_close(desc_c, desc, low_tolerance=True)

    @pytest.mark.parametrize("map_dtype", [torch.float32, torch.float64])
    def test_use_triton_true_raises_where_the_kernels_do_not_run(self, device, map_dtype):
        if device.type == "cuda" and map_dtype == torch.float32 and _HAS_TRITON:
            pytest.skip("the kernels run here")
        if device.type == "mps" and map_dtype == torch.float64:
            pytest.skip("MPS has no float64")
        net = CLIDDModel("A48", use_triton=True).to(device, map_dtype).eval()
        maps, _ = net(torch.rand(1, 3, 32, 32, device=device, dtype=map_dtype))
        with pytest.raises(BaseError, match="use_triton=True"):
            net.describe(maps, torch.zeros(1, 1, 2, device=device, dtype=map_dtype))

    def test_describe_no_keypoints(self, device, dtype):
        net = CLIDDModel("A48").to(device, dtype).eval()
        maps, _ = net(torch.rand(2, 3, 32, 32, device=device, dtype=dtype))
        desc = net.describe(maps, torch.zeros(2, 0, 2, device=device, dtype=dtype))
        assert desc.shape == (2, 0, 48)
        assert desc.dtype == dtype

    def test_describe_no_keypoints_backpropagates(self, device):
        net = CLIDDModel("A48", use_triton=False).to(device).eval()
        img = torch.rand(1, 3, 32, 32, device=device, requires_grad=True)
        maps, _ = net(img)
        net.describe(maps, torch.zeros(1, 0, 2, device=device)).sum().backward()
        assert img.grad is not None
        assert (img.grad == 0).all()

    @pytest.mark.parametrize("variant", ["A48", "L64", "G128", "U128"])
    def test_automatic_mode_runs_triton_for_the_128_dimensional_variants_on_cuda(self, variant, device):
        x = torch.zeros(1, device=device)
        expected = variant in ("G128", "E128", "U128") and x.is_cuda and _HAS_TRITON and not torch_version_lt(2, 7, 0)
        assert clidd_module._uses_triton(None, variant, x) is expected
        assert clidd_module._uses_triton(False, variant, x) is False


@pytest.mark.applies_to(
    devices=("cuda",),
    dtypes=("float32", "float16", "bfloat16"),
    reason="the Triton kernels run on CUDA, for float32, float16 and bfloat16",
)
class TestCLIDDTriton(BaseTester):
    @pytest.mark.parametrize("variant", ["A48", "M64", "G128"])
    def test_triton_matches_pytorch(self, variant, device, dtype):
        if not _HAS_TRITON:
            pytest.skip("triton is not installed")
        torch.manual_seed(0)
        net = CLIDDModel(variant, use_triton=False).to(device, dtype).eval()
        fused = CLIDDModel(variant, use_triton=True).to(device, dtype).eval()
        fused.load_state_dict(net.state_dict())
        with torch.no_grad():
            maps, _ = net(torch.rand(2, 3, 64, 96, device=device, dtype=dtype))
        maps = tuple(m.detach().requires_grad_() for m in maps)
        # Points beyond the image test the zero padding.
        kp = torch.rand(2, 333, 2, device=device, dtype=dtype) * 2.4 - 1.2
        out = net.describe(maps, kp)
        out_fused = fused.describe(maps, kp)
        self.assert_close(out_fused, out, low_tolerance=dtype not in (torch.float32, torch.float64))

        weights = torch.randn_like(out)
        grads = torch.autograd.grad((out * weights).sum(), [*maps, net.dcn.weight, net.post_conv.bias])
        grads_fused = torch.autograd.grad((out_fused * weights).sum(), [*maps, fused.dcn.weight, fused.post_conv.bias])
        # Hold both paths to a float64 reference, relative to the largest gradient. Both CUDA backward passes add
        # with atomics in a run-dependent order, and a pixel of the 1/32 map collects hundreds of contributions, so the
        # two float32 paths can differ by a few 1e-5 in one element from run to run; both stay about 1e-6 of the largest
        # gradient away from float64. In half precision they are up to about 5 % (bfloat16, G128 bias) and 4 %
        # (float16, M64 offset weights, the PyTorch path) away; an indexing or sign error would be of the order of the
        # gradient itself.
        ref = CLIDDModel(variant, use_triton=False).to(device, torch.float64).eval()
        ref.load_state_dict(net.state_dict())
        maps64 = tuple(m.detach().double().requires_grad_() for m in maps)
        out64 = ref.describe(maps64, kp.double())
        grads64 = torch.autograd.grad((out64 * weights.double()).sum(), [*maps64, ref.dcn.weight, ref.post_conv.bias])
        tol = {torch.float32: 1e-4, torch.float16: 0.05, torch.bfloat16: 0.15}[dtype]
        for g_fused, g, g64 in zip(grads_fused, grads, grads64):
            scale = g64.abs().max()
            assert (g.double() - g64).abs().max() <= tol * scale
            assert (g_fused.double() - g64).abs().max() <= tol * scale

    def test_triton_dynamo(self, device, torch_optimizer):
        if not _HAS_TRITON:
            pytest.skip("triton is not installed")
        # G128 has several output blocks per kernel launch. Before torch 2.7 the kernels stay out of the graph.
        torch.manual_seed(0)
        net = CLIDDModel("G128", use_triton=True).to(device).eval()
        x = torch.rand(1, 3, 96, 128, device=device)
        kp = torch.rand(1, 300, 2, device=device) * 2 - 1
        weights = torch.randn(1, 300, 128, device=device)

        def fn(img, k):
            maps, _ = net(img)
            return net.describe(maps, k)

        x_compiled, x_eager = x.clone().requires_grad_(), x.clone().requires_grad_()
        out_compiled = torch_optimizer(fn)(x_compiled, kp)
        out_eager = fn(x_eager, kp)
        self.assert_close(out_compiled, out_eager)
        # A weighted sum: the descriptors are unit vectors, so the gradient of a sum of their squares is rounding noise.
        (out_compiled * weights).sum().backward()
        (out_eager * weights).sum().backward()
        # The backward adds with atomics in a run-dependent order (see test_triton_matches_pytorch): compare relative
        # to the largest gradient, about 1e-6 apart.
        assert (x_compiled.grad - x_eager.grad).abs().max() <= 1e-4 * x_eager.grad.abs().max()

    def test_clidd_empty_keypoints_with_triton(self, device):
        if not _HAS_TRITON:
            pytest.skip("triton is not installed")
        clidd = _random_clidd("A48", device, torch.float32, detection_threshold=1e9, use_triton=True)
        out = clidd(_image().to(device))[0]
        assert out["keypoints"].shape == (0, 2)
        assert out["descriptors"].shape == (0, 48)


class TestCLIDD(BaseTester):
    def test_smoke(self, device, dtype):
        clidd = _random_clidd("A48", device, dtype)
        with torch.no_grad():
            out = clidd(_image(2).to(device, dtype))
        assert len(out) == 2
        for feats in out:
            n = feats["keypoints"].shape[0]
            assert feats["keypoints"].shape == (n, 2)
            assert feats["scores"].shape == (n,)
            assert feats["descriptors"].shape == (n, 48)
            assert feats["descriptors"].dtype == dtype
            assert feats["keypoints"].dtype == torch.promote_types(dtype, torch.float32)

    def test_empty_batch(self, device):
        clidd = _random_clidd("A48", device, torch.float32)
        assert clidd(torch.rand(0, 3, 64, 64, device=device)) == []

    def test_no_keypoint_above_the_threshold(self, device):
        clidd = _random_clidd("A48", device, torch.float32, detection_threshold=1e9)
        out = clidd(_image().to(device))[0]
        assert out["keypoints"].shape == (0, 2)
        assert out["scores"].shape == (0,)
        assert out["descriptors"].shape == (0, 48)

    def test_exception(self, device):
        clidd = _random_clidd("A48", device, torch.float32)
        with pytest.raises(BaseError):
            clidd(torch.rand(1, 2, 64, 64, device=device))
        with pytest.raises(BaseError):
            CLIDD("A48", top_k=0)

    def test_top_k_and_descending_scores(self, device):
        clidd = _random_clidd("A48", device, torch.float32, top_k=None)
        out_all = clidd(_image().to(device))[0]
        clidd.top_k = 10
        out = clidd(_image().to(device))[0]
        assert out_all["scores"].shape[0] > 10
        assert out["keypoints"].shape == (10, 2)
        assert (out_all["scores"][1:] <= out_all["scores"][:-1]).all()
        self.assert_close(out["scores"], out_all["scores"][:10])
        self.assert_close(out["keypoints"], out_all["keypoints"][:10])

    def test_descriptors_are_normalised(self, device, dtype):
        clidd = _random_clidd("A48", device, dtype)
        with torch.no_grad():
            desc = clidd(_image().to(device, dtype))[0]["descriptors"]
        self.assert_close(desc.float().norm(dim=-1), torch.ones(desc.shape[0], device=device), low_tolerance=True)

    def test_each_image_is_described_independently_of_the_batch(self, device):
        clidd = _random_clidd("A48", device, _wide(device))
        # A random image: the periodic _image() gives tied scores, whose top-k order is not defined.
        imgs = torch.rand(2, 3, 64, 96, device=device, dtype=_wide(device))
        batch = clidd(imgs)
        for b in range(2):
            single = clidd(imgs[b : b + 1])[0]
            for key in ("keypoints", "scores", "descriptors"):
                self.assert_close(batch[b][key], single[key])

    def test_grayscale_is_repeated_to_rgb(self, device):
        clidd = _random_clidd("A48", device, torch.float32)
        gray = _image()[:, :1].to(device)
        out_gray = clidd(gray)[0]
        out_rgb = clidd(gray.repeat(1, 3, 1, 1))[0]
        for key in ("keypoints", "scores", "descriptors"):
            self.assert_close(out_gray[key], out_rgb[key])

    def test_descriptors_are_differentiable(self, device):
        clidd = _random_clidd("A48", device, torch.float32, use_triton=False)
        img = _image().to(device).requires_grad_()
        clidd(img)[0]["descriptors"].sum().backward()
        assert img.grad is not None
        assert img.grad.abs().sum() > 0

    def test_convention_clidd_keypoints_are_strict_window_maxima_away_from_the_border(self, device):
        clidd = _random_clidd("A48", device, torch.float32, top_k=None)
        img = torch.rand(1, 3, 96, 128, device=device)
        out = clidd(img)[0]
        _, score = clidd.net(img)
        score = score[0, 0]
        assert out["keypoints"].shape[0] > 0
        x, y = out["keypoints"].long().unbind(-1)
        assert (x >= 4).all()
        assert (x < 128 - 4).all()
        assert (y >= 4).all()
        assert (y < 96 - 4).all()
        self.assert_close(score[y, x], out["scores"])
        assert (out["scores"] > clidd.detection_threshold).all()
        for xi, yi in zip(x.tolist(), y.tolist()):
            window = score[yi - 2 : yi + 3, xi - 2 : xi + 3].flatten()
            others = torch.cat([window[:12], window[13:]])
            assert (score[yi, xi] > others).all()

    def test_convention_clidd_tied_maxima_are_not_keypoints(self, device, monkeypatch):
        """Two equal maxima in one window are both suppressed; the reference keeps both."""
        clidd = _random_clidd("A48", device, torch.float32, top_k=None)
        img = torch.rand(1, 3, 64, 64, device=device)
        maps, _ = clidd.net(img)
        score = torch.full((1, 1, 64, 64), -10.0, device=device)
        score[0, 0, 20, 20] = score[0, 0, 20, 22] = 1.0  # a tie, 2 px apart
        score[0, 0, 40, 40] = 2.0  # a strict maximum
        monkeypatch.setattr(clidd.net, "forward", lambda x: (maps, score))
        out = clidd(img)[0]
        self.assert_close(out["keypoints"], torch.tensor([[40.0, 40.0]], device=device))
        self.assert_close(out["scores"], torch.tensor([2.0], device=device))

    def test_no_keypoints_backpropagates(self, device):
        clidd = _random_clidd("A48", device, torch.float32, detection_threshold=1e9, use_triton=False)
        img = _image().to(device).requires_grad_()
        clidd(img)[0]["descriptors"].sum().backward()
        assert img.grad is not None
        assert (img.grad == 0).all()

    def test_convention_clidd_reports_the_pixel_centre_of_the_resize(self, device):
        """A 70 x 100 image is resized to 64 x 96: pixel i of the resize is reported at (i + 0.5) W / W' - 0.5."""
        wide = _wide(device)
        clidd = _random_clidd("A48", device, wide, top_k=None)
        img = torch.rand(1, 3, 70, 100, device=device, dtype=wide)
        kpts = clidd(img)[0]["keypoints"]
        resized = F.interpolate(img, (64, 96), mode="bilinear", align_corners=False)
        kpts_resized = clidd(resized)[0]["keypoints"]
        assert kpts_resized.shape[0] > 0
        # The 64 x 96 image is not resized again, so its keypoints are its integer pixels.
        self.assert_close(kpts_resized, kpts_resized.round())
        ratio = torch.tensor([100 / 96, 70 / 64], device=device, dtype=wide)
        self.assert_close(kpts, (kpts_resized + 0.5) * ratio - 0.5)
        # That is the input position the resize reads: a ramp holding each pixel's own x (y) coordinate, resized,
        # holds (i + 0.5) W / W' - 0.5 at pixel i away from the clamped border.
        ys, xs = torch.meshgrid(
            torch.arange(70, device=device, dtype=wide), torch.arange(100, device=device, dtype=wide), indexing="ij"
        )
        ramp = F.interpolate(torch.stack([xs, ys])[None], (64, 96), mode="bilinear", align_corners=False)[0]
        i, j = torch.arange(1, 95, device=device), torch.arange(1, 63, device=device)
        self.assert_close(ramp[0, 10, i], (i.to(wide) + 0.5) * 100 / 96 - 0.5)
        self.assert_close(ramp[1, j, 10], (j.to(wide) + 0.5) * 70 / 64 - 0.5)

    @pytest.mark.applies_to(devices=("cpu",), reason="checks the call to the downloader, which is device independent")
    def test_pretrained_uses_kornia_downloader(self, monkeypatch, device):
        calls = []

        def fake_download(url, **kwargs):
            calls.append((url, kwargs))
            return CLIDDModel("A48").state_dict()

        monkeypatch.setattr(clidd_module, "load_state_dict_from_url", fake_download)
        model = CLIDD.from_pretrained("A48")
        assert calls == [
            (
                "https://github.com/HITCSC/CLIDD/releases/download/v1.0/A48.pth",
                {"file_name": "clidd_A48.pth", "map_location": "cpu"},
            )
        ]
        assert not model.training

    @pytest.mark.slow
    def test_pretrained_a48_matches_the_reference_implementation(self, device):
        # Generated with the authors' code (github.com/HITCSC/CLIDD @ 0a89201a, weights v1.0 A48.pth) on CUDA with
        # TRITON_F32_DEFAULT=ieee and cuDNN TF32 off, top_k=None, for _image(), sorted by descending score. No 5 x 5
        # window of its score map holds a tie, so kornia's strict NMS keeps the same 215 keypoints:
        #     ref = CLIDD("A48", top_k=None).cuda().eval()   # the authors' clidd.CLIDD
        #     r = ref(_image().cuda())[0]; order = r["scores"].argsort(descending=True)
        #     r["keypoints"][order][:6], r["scores"][order][:6], r["descriptors"][order][:3, :6]
        clidd = CLIDD.from_pretrained("A48", top_k=None).to(device)
        with torch.no_grad():
            out = clidd(_image().to(device))[0]
        assert out["keypoints"].shape[0] == 215
        expected_kpts = torch.tensor(
            [[103.0, 85.0], [69.0, 51.0], [69.0, 17.0], [103.0, 51.0], [86.0, 51.0], [35.0, 51.0]], device=device
        )
        expected_scores = torch.tensor(
            [
                1.4006505012512207,
                1.3772854804992676,
                1.014133095741272,
                0.60288006067276,
                0.46606725454330444,
                0.16981136798858643,
            ],
            device=device,
        )
        expected_desc = torch.tensor(
            [
                [
                    0.26609036326408386,
                    0.06286061555147171,
                    -0.10836666077375412,
                    0.044484008103609085,
                    -0.05933245271444321,
                    -0.22280138731002808,
                ],
                [
                    0.17852981388568878,
                    0.25658389925956726,
                    0.16324763000011444,
                    -0.015240903943777084,
                    0.0389375202357769,
                    0.006418473087251186,
                ],
                [
                    0.07993254065513611,
                    0.0385250523686409,
                    0.23364156484603882,
                    0.026453522965312004,
                    0.0455319806933403,
                    -0.00934551376849413,
                ],
            ],
            device=device,
        )
        self.assert_close(out["keypoints"][:6], expected_kpts, rtol=0, atol=0)
        self.assert_close(out["scores"][:6], expected_scores, rtol=1e-4, atol=1e-4)
        self.assert_close(out["descriptors"][:3, :6], expected_desc, rtol=1e-4, atol=1e-4)


class TestCLIDDMatch(BaseTester):
    def test_match_is_mutual_dual_softmax(self, device, dtype):
        torch.manual_seed(0)
        wide = _wide(device)
        d0 = F.normalize(torch.randn(30, 16, device=device, dtype=wide), dim=-1)
        d1 = F.normalize(
            torch.cat(
                [
                    d0[:20] + 0.05 * torch.randn(20, 16, device=device, dtype=wide),
                    torch.randn(15, 16, device=device, dtype=wide),
                ]
            ),
            dim=-1,
        )
        scores, idxs = CLIDD.match(d0.to(dtype), d1.to(dtype), min_score=0.0)
        prob = torch.softmax(20 * d0 @ d1.T, 1) * torch.softmax(20 * d0 @ d1.T, 0)
        nn01, nn10 = prob.argmax(1), prob.argmax(0)
        mutual = torch.nonzero(nn10[nn01] == torch.arange(30, device=device))[:, 0]
        expected = torch.stack([mutual, nn01[mutual]], -1)
        if dtype in (torch.float32, torch.float64):
            # In half precision, near-ties among the unrelated descriptors can pick another neighbour.
            assert idxs.tolist() == expected.tolist()
            self.assert_close(scores[:, 0], prob[mutual, nn01[mutual]].to(dtype))
        # The noisy copies are matched to their originals.
        assert {(i, i) for i in range(20)} <= {tuple(p) for p in idxs.tolist()}

    @pytest.mark.parametrize(("similarity", "beta"), [(0.5, 20.0), (-1.0, 30.0)])
    def test_singleton_pair_scores_one(self, device, dtype, similarity, beta):
        """Each softmax of a 1 x 1 similarity is 1, however low the similarity: the score does not underflow."""
        d0 = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        d1 = torch.tensor([[similarity, (1 - similarity**2) ** 0.5]], device=device, dtype=dtype)
        scores, idxs = CLIDD.match(d0, d1, beta=beta)
        assert idxs.tolist() == [[0, 0]]
        self.assert_close(scores, torch.ones(1, 1, device=device, dtype=dtype))
        assert scores.dtype == dtype

    def test_unmatched_descriptor_keeps_the_other_matches(self, device, dtype):
        """A descriptor of desc0 far from every one of desc1 must not drop the matches of the others."""
        d = torch.eye(4, device=device, dtype=dtype)
        d0 = torch.cat([d[:3], -d[:1]])  # the last is opposite to desc1[0] and orthogonal to the rest
        scores, idxs = CLIDD.match(d0, d[:3])
        assert idxs.tolist() == [[0, 0], [1, 1], [2, 2]]
        assert torch.isfinite(scores).all()

    def test_min_score(self, device):
        torch.manual_seed(0)
        d0 = F.normalize(torch.randn(30, 16, device=device), dim=-1)
        d1 = F.normalize(torch.randn(25, 16, device=device), dim=-1)
        scores_all, _ = CLIDD.match(d0, d1, min_score=0.0)
        scores, _ = CLIDD.match(d0, d1, min_score=0.2)
        assert (scores > 0.2).all()
        assert scores.shape[0] == (scores_all > 0.2).sum()

    def test_empty(self, device):
        scores, idxs = CLIDD.match(torch.zeros(0, 8, device=device), torch.rand(5, 8, device=device))
        assert scores.shape == (0, 1)
        assert idxs.shape == (0, 2)
        assert idxs.dtype == torch.long
