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

from kornia.feature import DeFMO
from kornia.feature.defmo import RenderingDeFMO

from testing.base import BaseTester


class TestRenderingDeFMOTimesBuffer(BaseTester):
    """`times` must be a real buffer so `.to()` moves it, and `forward` must not rebind it.

    Same bug shape as #4069/#4079 (SIFTDescriptor.gk et al.) in a different class: `times`
    was a plain Python attribute, invisible to `Module.to()`/`.half()`, and `forward`
    compensated by re-deriving its DEVICE (but never its dtype) from the input on every
    call -- so `.half()` a module and calling it crashed with a dtype-mismatch RuntimeError.
    """

    def test_to_moves_times(self, device):
        # float16 rather than the default float32, or the dtype assertion would hold
        # vacuously; float16 also works on MPS, where float64 is unavailable.
        mod = RenderingDeFMO().to(device, torch.float16)
        assert mod.times.dtype == torch.float16
        assert mod.times.device == torch.empty(0, device=device).device

    def test_times_stays_out_of_state_dict(self, device):
        # non-persistent: `times` is fully determined by `tsr_steps`, not learned, so
        # existing checkpoints keep loading with strict=True.
        assert "times" not in RenderingDeFMO().state_dict()

    def test_forward_does_not_mutate_the_module(self, device):
        mod = RenderingDeFMO().to(device).eval()
        before = (mod.times.dtype, mod.times.device)
        latent = torch.rand(1, 2048, 4, 4, device=device)
        with torch.no_grad():
            mod(latent)
        assert (mod.times.dtype, mod.times.device) == before

    @pytest.mark.slow
    def test_half_precision_forward_no_longer_crashes(self, device):
        # the actual reported defect: .half() left `times` at float32 (only device was
        # ever re-derived in forward, never dtype), so a half-precision forward crashed
        # with "Input type ... and weight type ... should be the same".
        mod = RenderingDeFMO().to(device, torch.float16).eval()
        latent = torch.rand(1, 2048, 4, 4, device=device, dtype=torch.float16)
        with torch.no_grad():
            out = mod(latent)
        assert out.dtype == torch.float16

    def test_matches_pre_fix_output_in_the_normal_float32_path(self, device):
        # behaviour-preserving: pinned against the pre-fix module (plain `self.times`
        # attribute, `.to(latent.device)`-only) with the same seed and input -- confirmed
        # byte-identical (torch.equal) locally before this value was hardcoded here.
        # A tight absolute tolerance doesn't survive kornia's own CI matrix, though:
        # different torch versions/BLAS backends on "cpu" still reorder the Bottleneck
        # blocks' float32 accumulation differently (observed 201292.0 vs 201291.984375,
        # a ~1.6e-8 relative difference -- ordinary floating-point non-associativity, not
        # a behavioral change). rel=1e-3 keeps ~1000x margin above that while still
        # catching a real regression, which would produce a very different value.
        if device.type != "cpu":
            pytest.skip("checksum pinned on CPU; cross-device float summation can differ further")
        torch.manual_seed(0)
        mod = RenderingDeFMO().to(device).eval()
        latent = torch.rand(1, 2048, 4, 4, device=device)
        with torch.no_grad():
            out = mod(latent)
        assert out.shape == (1, 24, 4, 64, 64)
        assert out.dtype == torch.float32
        assert out.sum().item() == pytest.approx(201291.984375, rel=1e-3)


class TestDeFMO(BaseTester):
    @pytest.mark.slow
    def test_shape(self, device, dtype):
        inp = torch.ones(1, 6, 128, 160, device=device, dtype=dtype)
        defmo = DeFMO(pretrained=False).to(device, dtype)
        defmo.eval()  # batchnorm with size 1 is not allowed in train mode
        out = defmo(inp)
        assert out.shape == (1, 24, 4, 128, 160)

    @pytest.mark.slow
    def test_shape_batch(self, device, dtype):
        inp = torch.ones(2, 6, 128, 160, device=device, dtype=dtype)
        defmo = DeFMO(pretrained=False).to(device, dtype).eval()
        out = defmo(inp)
        with torch.no_grad():
            assert out.shape == (2, 24, 4, 128, 160)

    @pytest.mark.slow
    def test_gradcheck(self, device):
        patches = torch.rand(2, 6, 64, 64, device=device, dtype=torch.float64)
        defmo = DeFMO(pretrained=False).to(patches.device, patches.dtype).eval()
        self.gradcheck(defmo, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8)

    @pytest.mark.slow
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 6, 128, 160
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = DeFMO(True).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(DeFMO(True).to(patches.device, patches.dtype).eval())
        with torch.no_grad():
            self.assert_close(model(patches), model_jit(patches))


class _RecordingRenderer(torch.nn.Module):
    """A stand-in for the rendering network: returns fixed values per output channel and records each input."""

    def __init__(self, values):
        super().__init__()
        self.values = values
        self.seen = []

    def forward(self, latent):
        self.seen.append(latent)
        b, _, h, w = latent.shape
        out = self.values.to(latent).view(1, -1, 1, 1).expand(b, -1, 16 * h, 16 * w)
        return out.clone()


class TestConventionsDeFMO(BaseTester):
    def test_convention_defmo_renders_24_rgba_subframes_in_time_order(self, device, dtype):
        # DeFMO takes (B, 6, H, W) and returns (B, 24, 4, H, W) for H and W multiples of 16; other sizes are rounded up
        # to the next multiple of 16. The encoder's latent is (B, 2048, H / 16, W / 16); sub-frame k is rendered from it
        # with the time t = k / 23 prepended as channel 0, and all four channels, RGB and alpha, go through a sigmoid.
        # The rendering network is replaced by a recorder that upsamples by 16 as the real one does. The input channel
        # order (the blurred RGB image, then the RGB background: the order of rozumden/DeFMO 9b69769 run.py) only
        # matters to the trained weights and is not visible here.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = DeFMO(pretrained=False).eval()
        model = model.to(device, dtype)
        values = torch.tensor([-3.0, -1.0, 0.5, 4.0], device=device, dtype=dtype)
        model.rendering.net = _RecordingRenderer(values)
        with torch.no_grad():
            out = model(torch.rand(2, 6, 64, 96, device=device, dtype=dtype))
            assert model(torch.rand(1, 6, 60, 90, device=device, dtype=dtype)).shape == (1, 24, 4, 64, 96)
        assert out.shape == (2, 24, 4, 64, 96)
        self.assert_close(out[1, 7, :, 50, 11], torch.sigmoid(values))
        self.assert_close(out[0, 23, :, 3, 90], torch.sigmoid(values))
        assert len(model.rendering.net.seen) == 48  # 24 sub-frames per call
        latents = model.rendering.net.seen[:24]
        assert all(latent.shape == (2, 2049, 4, 6) for latent in latents)
        times = torch.stack([latent[:, 0] for latent in latents])  # (24, B, h, w), each constant
        expected = torch.linspace(0, 1, 24, device=device, dtype=dtype).view(24, 1, 1, 1).expand_as(times)
        self.assert_close(times, expected)
