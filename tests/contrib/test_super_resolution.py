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

from kornia.contrib.super_resolution import (
    RRDBNetBuilder,
    SmallSRBuilder,
    SuperResolution,
    SuperResolutionConfig,
)

from testing.base import BaseTester


class TestSuperResolutionBuilders(BaseTester):
    """Pin the public super-resolution entry points.

    Regression test for #4291: ``SuperResolution`` did not implement ``ModelBase``'s
    abstract ``from_config`` and defined no ``__init__``, so every builder raised at
    construction. Nothing in ``tests/`` called the builders, which is why CI never
    noticed.
    """

    def test_small_sr_builder_builds_and_runs(self, device, dtype):
        model = SmallSRBuilder.build(pretrained=False).to(device, dtype)
        out = model(torch.rand(1, 3, 16, 16, device=device, dtype=dtype))
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    def test_rrdbnet_builder_builds_and_runs(self, device, dtype):
        model = RRDBNetBuilder.build(pretrained=False).to(device, dtype)
        out = model(torch.rand(1, 3, 16, 16, device=device, dtype=dtype))
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    @pytest.mark.parametrize("model_name", ["small_sr", "RealESRNet_x4plus"])
    def test_from_config_dispatches_to_both_families(self, device, dtype, model_name):
        config = SuperResolutionConfig(model_name=model_name, pretrained=False)
        model = SuperResolution.from_config(config).to(device, dtype)
        out = model(torch.rand(1, 3, 16, 16, device=device, dtype=dtype))
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    def test_from_config_rejects_unknown_model_name(self):
        with pytest.raises(ValueError, match="small_sr"):
            SuperResolution.from_config(SuperResolutionConfig(model_name="not_a_model", pretrained=False))

    @pytest.mark.parametrize("builder", [SmallSRBuilder, RRDBNetBuilder])
    def test_onnx_metadata_is_initialised(self, builder):
        """``to_onnx`` reads these three; without defaults they are annotations only.

        Regression test for the review of #4335: an ``__init__`` that does not set them
        leaves export raising ``AttributeError``.
        """
        model = builder.build(pretrained=False)
        assert model.pseudo_image_size is None or isinstance(model.pseudo_image_size, int)
        assert model.input_image_size is None or isinstance(model.input_image_size, int)
        assert model.output_image_size is None or isinstance(model.output_image_size, int)

    def test_rrdbnet_onnx_metadata_defaults_to_none(self):
        """``RRDBNetBuilder`` sets none of the three, so they must default to ``None``."""
        model = RRDBNetBuilder.build(pretrained=False)
        assert model.input_image_size is None
        assert model.output_image_size is None
        assert model.pseudo_image_size is None

    @pytest.mark.parametrize("builder", [SmallSRBuilder, RRDBNetBuilder])
    def test_to_onnx_exports(self, builder):
        """Both families export; this fails with ``AttributeError`` if the metadata is unset."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        builder.build(pretrained=False).to_onnx(save=False)

    def test_small_sr_output_image_size_uses_upscale_factor(self):
        """``output_image_size`` follows the documented ``image_size * upscale_factor``."""
        model = SmallSRBuilder.build(pretrained=False, upscale_factor=2, image_size=32)
        assert model.output_image_size == 64
