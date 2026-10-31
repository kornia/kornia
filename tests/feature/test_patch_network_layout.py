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
from torch import nn

import kornia.feature._normalization
import kornia.feature.affine_shape
import kornia.feature.hardnet
import kornia.feature.hynet
import kornia.feature.orientation
import kornia.feature.tfeat
from kornia.feature import HardNet, HardNet8, HyNet, LAFAffNetShapeEstimator, OriNet, SOSNet, TFeat
from kornia.feature._layout import _patch_conv_layout
from kornia.feature.laf import laf_from_center_scale_ori

from testing.base import BaseTester

NETWORKS = {
    "HardNet": HardNet,
    "HardNet8": HardNet8,
    "SOSNet": SOSNet,
    "TFeat": TFeat,
    "HyNet": HyNet,
    "AffNet": LAFAffNetShapeEstimator,
    "OriNet": OriNet,
}

# Every module that calls the helper imports it by name.
CALLERS = [
    kornia.feature._normalization,
    kornia.feature.affine_shape,
    kornia.feature.hardnet,
    kornia.feature.hynet,
    kornia.feature.orientation,
    kornia.feature.tfeat,
]


def _run(name, net, device, dtype):
    if name == "AffNet":
        img = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        centers = torch.tensor([[[20.0, 24.0], [40.0, 36.0], [31.0, 30.0]]], device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(centers, torch.full((1, 3, 1, 1), 6.0, device=device, dtype=dtype))
        return net(lafs, img)
    return net(torch.rand(3, 1, 32, 32, device=device, dtype=dtype))


class TestPatchConvLayout(BaseTester):
    def test_cpu_float32_follows_onednn(self):
        x = torch.rand(2, 4, 3, 3)
        converted = _patch_conv_layout(x, cpu_has_onednn=True)
        assert converted.is_contiguous(memory_format=torch.channels_last)
        assert not converted.is_contiguous()
        assert _patch_conv_layout(x, cpu_has_onednn=False) is x

    @pytest.mark.parametrize("other", [torch.float64, torch.float16, torch.bfloat16])
    def test_cpu_other_dtypes_keep_layout(self, other):
        x = torch.rand(2, 4, 3, 3, dtype=other)
        assert _patch_conv_layout(x, cpu_has_onednn=True) is x

    def test_values_unchanged(self, device, dtype):
        x = torch.rand(2, 4, 3, 3, device=device, dtype=dtype)
        assert torch.equal(_patch_conv_layout(x), x)

    @pytest.mark.parametrize("name", NETWORKS)
    def test_convolutions_receive_device_layout(self, name, device, dtype):
        torch.manual_seed(0)
        net = NETWORKS[name](pretrained=False).eval().to(device, dtype)
        probe = torch.empty(1, 2, 3, 3, device=device, dtype=dtype)
        channels_last = _patch_conv_layout(probe) is not probe
        convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
        seen: list[tuple[str, bool, bool]] = []

        def record(module, args):
            x = args[0]
            seen.append((repr(module), x.is_contiguous(memory_format=torch.channels_last), x.is_contiguous()))

        hooks = [conv.register_forward_pre_hook(record) for conv in convs]
        try:
            with torch.no_grad():
                _run(name, net, device, dtype)
        finally:
            for hook in hooks:
                hook.remove()
        assert len(seen) == len(convs)
        # A single-channel input is contiguous in both layouts, so the first convolution says nothing; every later one
        # must see the layout the helper picks for the device.
        for conv, is_channels_last, is_contiguous in seen[1:]:
            assert (is_channels_last and not is_contiguous) == channels_last, conv
            assert is_contiguous != channels_last, conv

    @pytest.mark.parametrize("name", NETWORKS)
    def test_layout_keeps_outputs(self, name, device, dtype, monkeypatch):
        torch.manual_seed(0)
        net = NETWORKS[name](pretrained=False).eval().to(device, dtype)
        torch.manual_seed(1)
        with torch.no_grad():
            expected = _run(name, net, device, dtype)
        for module in CALLERS:
            monkeypatch.setattr(module, "_patch_conv_layout", lambda x: x)
        torch.manual_seed(1)
        with torch.no_grad():
            actual = _run(name, net, device, dtype)
        self.assert_close(actual, expected)
