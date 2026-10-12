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

from kornia.core.check import ShapeError
from kornia.feature import HardNet, HardNet8, HyNet, MKDDescriptor, SIFTDescriptor, SOSNet, TFeat

from testing.base import BaseTester, supports_conv2d, supports_replicate_padding


class TestHardNet(BaseTester):
    @pytest.mark.slow
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        hardnet = HardNet().to(device)
        hardnet.eval()  # batchnorm with size 1 is not allowed in train mode
        out = hardnet(inp)
        assert out.shape == (1, 128)

    @pytest.mark.slow
    def test_shape_batch(self, device):
        inp = torch.ones(4, 1, 32, 32, device=device)
        hardnet = HardNet().to(device)
        out = hardnet(inp)
        assert out.shape == (4, 128)

    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float64)
        hardnet = HardNet().to(patches.device, patches.dtype)
        self.gradcheck(hardnet, (patches,), eps=1e-4, atol=1e-4, nondet_tol=1e-8)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = HardNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(HardNet().to(patches.device, patches.dtype).eval())
        self.assert_close(model(patches), model_jit(patches))


class TestHardNet8(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 1, 32, 32, device=device)
        hardnet = HardNet8().to(device)
        hardnet.eval()  # batchnorm with size 1 is not allowed in train mode
        out = hardnet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(4, 1, 32, 32, device=device)
        hardnet = HardNet8().to(device)
        out = hardnet(inp)
        assert out.shape == (4, 128)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float32)
        hardnet = HardNet8().to(patches.device, patches.dtype)
        self.gradcheck(hardnet, (patches,), eps=1e-4, atol=1e-4)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = HardNet8().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(HardNet8().to(patches.device, patches.dtype).eval())
        self.assert_close(model(patches), model_jit(patches))

    def test_untrained_descriptors_are_distinct_5692(self, device, dtype):
        # An all-ones PCA placeholder mapped every patch to +-(1, ..., 1) / sqrt(128).
        if not supports_conv2d(device, dtype):
            pytest.skip(f"no conv2d kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        patches = torch.rand(8, 1, 32, 32, device=device, dtype=dtype)
        out = HardNet8().to(device, dtype)(patches)
        assert out.shape == (8, 128)
        self.assert_close(out.norm(dim=1), torch.ones(8, device=device, dtype=dtype))
        cos = out @ out.T
        off_diagonal = cos[~torch.eye(8, dtype=torch.bool, device=device)]
        assert off_diagonal.abs().max() < 0.99

    def test_untrained_model_keeps_the_first_features_5692(self, device, dtype):
        if not supports_conv2d(device, dtype):
            pytest.skip(f"no conv2d kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        patches = torch.rand(4, 1, 32, 32, device=device, dtype=dtype)
        model = HardNet8().to(device, dtype)
        features = model.features(model._normalize_input(patches)).view(4, -1)
        expected = torch.nn.functional.normalize(torch.nn.functional.normalize(features, dim=1)[:, :128], dim=1)
        self.assert_close(model(patches), expected)

    def test_untrained_model_is_trainable_5692(self, device, dtype):
        # The all-ones placeholder made the normalised output constant, so the gradient was zero (~1e-15).
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision rounding leaves a noise gradient on the degenerate placeholder as well")
        torch.manual_seed(0)
        patches = torch.rand(4, 1, 32, 32, device=device, dtype=dtype)
        model = HardNet8().to(device, dtype)
        target = torch.randn(4, 128, device=device, dtype=dtype)
        (model(patches) * target).sum().backward()
        grad = model.features[0].weight.grad
        assert grad is not None
        assert grad.abs().max() > 1e-3

    def test_checkpoint_layout_is_unchanged_5692(self, device):
        # The pretrained checkpoint loads with strict=True, so the buffer names and shapes must stay.
        model = HardNet8().to(device)
        state = model.state_dict()
        assert state["components"].shape == (512, 128)
        assert state["mean"].shape == (512,)
        loaded = torch.rand(512, 128, device=device)
        state["components"] = loaded
        model.load_state_dict(state, strict=True)
        self.assert_close(model.components, loaded)


class TestHardNetConstantPatchIsFinite(BaseTester):
    """A constant patch drives the features to zero; the L2 normalisation must not give NaN.

    `F.normalize`'s default eps of 1e-12 rounds to zero in float16, so `0 / 0` reached the
    descriptor. A detector that pads a short result feeds exactly this patch through a zero LAF.
    """

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("model", [HardNet, HardNet8])
    def test_constant_patch(self, device, desc_dtype, model):
        if not supports_conv2d(device, desc_dtype):
            # The descriptor is a stack of convolutions; torch 2.1.2 has no float16 CPU kernel.
            pytest.skip(f"no conv2d kernel for {desc_dtype} on {device.type}")
        patches = torch.zeros(2, 1, 32, 32, device=device, dtype=desc_dtype)
        out = model().to(device, desc_dtype)(patches)
        assert torch.isfinite(out).all()


# Each patch descriptor with the weights it gets without a download, its patch size, its output width and whether it
# pads with mode="replicate" (its gradient filters). MKD runs unwhitened, which needs no download either and returns
# the full 238-dimensional concat embedding.
_PATCH_DESCRIPTORS = {
    "HardNet": (HardNet, 32, 128, False),
    "HardNet8": (HardNet8, 32, 128, False),
    "HyNet": (HyNet, 32, 128, False),
    "SOSNet": (SOSNet, 32, 128, False),
    "TFeat": (TFeat, 32, 128, False),
    "MKDDescriptor": (lambda: MKDDescriptor(whitening=None), 32, 238, True),
    "SIFTDescriptor": (lambda: SIFTDescriptor(32), 32, 128, True),
}


class TestConventionsPatchDescriptors(BaseTester):
    """Pins for the patch-descriptor family: the patch each takes, its output norm and its intensity-range rule."""

    @staticmethod
    def _model(name, device, dtype):
        factory, patch_size, dim, pads = _PATCH_DESCRIPTORS[name]
        if not supports_conv2d(device, dtype):
            pytest.skip(f"no conv2d kernel for {dtype} on {device.type}")
        if pads and not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        torch.manual_seed(0)  # random-init weights: no pin downloads
        return factory().to(device, dtype), patch_size, dim

    @pytest.mark.parametrize("name", ["HardNet", "HardNet8", "SOSNet", "TFeat"])
    def test_convention_learned_descriptors_take_one_channel_32x32_patches(self, device, dtype, name):
        # The input is validated: another size or channel count raises instead of returning another width. HyNet's
        # identical check is pinned by test_hynet.py::TestHyNet::test_wrong_patch_shape_is_rejected_5724.
        model, _, _ = self._model(name, device, dtype)
        assert type(model).patch_size == 32
        assert model(torch.rand(3, 1, 32, 32, device=device, dtype=dtype)).shape == (3, 128)
        for shape in [(3, 1, 40, 40), (3, 1, 64, 64), (3, 3, 32, 32), (3, 1, 32, 31)]:
            with pytest.raises(ShapeError):
                model(torch.rand(*shape, device=device, dtype=dtype))

    @pytest.mark.parametrize("name", ["HardNet", "HardNet8", "HyNet", "SOSNet", "MKDDescriptor", "SIFTDescriptor"])
    def test_convention_patch_descriptors_return_unit_norm_rows(self, device, dtype, name):
        # TFeat is the exception, a tanh output as in its reference: test_tfeat.py::TestConventionsTFeat.
        model, patch_size, dim = self._model(name, device, dtype)
        out = model(torch.rand(5, 1, patch_size, patch_size, device=device, dtype=dtype))
        assert out.shape == (5, dim)
        self.assert_close(out.norm(dim=1), torch.ones(5, device=device, dtype=dtype))

    @pytest.mark.parametrize("name", ["HardNet", "HardNet8", "SOSNet", "TFeat", "MKDDescriptor", "SIFTDescriptor"])
    def test_convention_patch_descriptors_ignore_an_affine_intensity_change(self, device, dtype, name):
        # HardNet and HardNet8 standardise each patch, SOSNet and TFeat start with InstanceNorm2d, MKD and SIFT use
        # gradients and an L2 normalisation, so a [0, 1] patch, 2x + 1 and 255x give one descriptor. TFeat keeps the
        # InstanceNorm eps (1e-5 on the variance) in its biased tanh layers, which moves it by ~1e-5 here.
        # HyNet is the exception (scale only): test_hynet.py::TestConventionsHyNet.
        model, patch_size, _ = self._model(name, device, dtype)
        patches = torch.rand(4, 1, patch_size, patch_size, device=device, dtype=dtype)
        desc = model(patches)
        tol = {"rtol": 0.0, "atol": 1e-4} if dtype in (torch.float32, torch.float64) else {}
        # A float16 InstanceNorm that accumulates the squares of a [0, 255] patch in float16 overflows (torch 2.5.1 on
        # CPU does), so the float16 leg scales by 16 instead.
        scale = 16.0 if dtype == torch.float16 else 255.0
        self.assert_close(model(2 * patches + 1), desc, **tol)
        self.assert_close(model(scale * patches), desc, **tol)
