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
        # byte-identical (torch.equal) before this value was hardcoded here.
        if device.type != "cpu":
            pytest.skip("checksum pinned on CPU; cross-device float summation can differ in the ULP")
        torch.manual_seed(0)
        mod = RenderingDeFMO().to(device).eval()
        latent = torch.rand(1, 2048, 4, 4, device=device)
        with torch.no_grad():
            out = mod(latent)
        assert out.shape == (1, 24, 4, 64, 64)
        assert out.dtype == torch.float32
        assert out.sum().item() == pytest.approx(201291.984375, abs=1e-3)


class TestDeFMO(BaseTester):
    @pytest.mark.slow
    def test_shape(self, device, dtype):
        inp = torch.ones(1, 6, 128, 160, device=device, dtype=dtype)
        defmo = DeFMO().to(device, dtype)
        defmo.eval()  # batchnorm with size 1 is not allowed in train mode
        out = defmo(inp)
        assert out.shape == (1, 24, 4, 128, 160)

    @pytest.mark.slow
    def test_shape_batch(self, device, dtype):
        inp = torch.ones(2, 6, 128, 160, device=device, dtype=dtype)
        defmo = DeFMO().to(device, dtype)
        out = defmo(inp)
        with torch.no_grad():
            assert out.shape == (2, 24, 4, 128, 160)

    @pytest.mark.slow
    def test_gradcheck(self, device):
        patches = torch.rand(2, 6, 64, 64, device=device, dtype=torch.float64)
        defmo = DeFMO().to(patches.device, patches.dtype)
        self.gradcheck(defmo, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8)

    @pytest.mark.slow
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 6, 128, 160
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = DeFMO(True).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(DeFMO(True).to(patches.device, patches.dtype).eval())
        with torch.no_grad():
            self.assert_close(model(patches), model_jit(patches))
