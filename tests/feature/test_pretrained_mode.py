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

"""The mode in which the learned models of ``kornia.feature`` are constructed.

A model built with complete pretrained weights comes back in evaluation mode; a randomly initialised one stays in
PyTorch's default training mode. The classes with a ``pretrained`` flag default to their pretrained weights.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from kornia.feature import (
    ALIKED,
    DISK,
    SOLD2,
    DeDoDe,
    DeFMO,
    HardNet,
    HardNet8,
    HyNet,
    KeyNet,
    KeyNetAffNetHardNet,
    KeyNetDetector,
    LAFAffNetShapeEstimator,
    LAFDescriptor,
    LightGlue,
    LoFTR,
    OriNet,
    SOLD2_detector,
    SOSNet,
    TFeat,
    XFeat,
)
from kornia.feature.sold2 import sold2, sold2_detector

from testing.base import BaseTester

# The classes whose constructor takes a ``pretrained`` flag.
_FLAG_CLASSES = [
    HardNet,
    HardNet8,
    HyNet,
    SOSNet,
    TFeat,
    OriNet,
    LAFAffNetShapeEstimator,
    KeyNet,
    KeyNetDetector,
    DeFMO,
]

# Constructions that load no weights, each of which must stay in training mode.
_RANDOM_INIT = [
    *[pytest.param(lambda cls=cls: cls(pretrained=False), id=cls.__name__) for cls in _FLAG_CLASSES],
    pytest.param(lambda: SOLD2(pretrained=False), id="SOLD2"),
    pytest.param(lambda: SOLD2_detector(pretrained=False), id="SOLD2_detector"),
    pytest.param(lambda: LoFTR(pretrained=None), id="LoFTR"),
    pytest.param(XFeat, id="XFeat"),
    pytest.param(lambda: ALIKED(model_name="aliked-n16"), id="ALIKED"),
    pytest.param(DISK, id="DISK", marks=pytest.mark.slow),
    # The B descriptor: the default G one downloads the DINOv2 backbone in its constructor.
    pytest.param(lambda: DeDoDe(descriptor_model="B"), id="DeDoDe", marks=pytest.mark.slow),
]

# Factories that load weights, each of which must return the model in evaluation mode.
_WEIGHTED = [
    pytest.param(SOLD2, id="SOLD2"),
    pytest.param(SOLD2_detector, id="SOLD2_detector"),
    pytest.param(lambda: LoFTR("outdoor"), id="LoFTR"),
    pytest.param(XFeat.from_pretrained, id="XFeat"),
    pytest.param(lambda: ALIKED.from_pretrained("aliked-n16"), id="ALIKED"),
    pytest.param(DISK.from_pretrained, id="DISK", marks=pytest.mark.slow),
    # The pair of the DeDoDe.from_pretrained doctest, which CI prefetches; the default G descriptor is 2.2 GB.
    pytest.param(
        lambda: DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="B-upright"),
        id="DeDoDe",
        marks=pytest.mark.slow,
    ),
]


def _network(model: nn.Module) -> nn.Module:
    """Return the learned network: ``KeyNetDetector`` holds its ``KeyNet`` as ``.model`` and sets no mode itself."""
    return model.model if isinstance(model, KeyNetDetector) else model


class TestPretrainedMode(BaseTester):
    @pytest.mark.parametrize("cls", _FLAG_CLASSES, ids=lambda cls: cls.__name__)
    def test_flag_classes_default_to_pretrained_and_eval(self, cls):
        assert inspect.signature(cls).parameters["pretrained"].default is True
        default = _network(cls())
        assert not default.training
        assert not any(m.training for m in default.modules())
        expected = _network(cls(pretrained=True)).state_dict()
        actual = default.state_dict()
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert torch.equal(actual[key], value), key

    @pytest.mark.parametrize("factory", _RANDOM_INIT)
    def test_random_init_is_training_mode(self, factory):
        model = _network(factory())
        assert model.training is True
        assert all(m.training for m in model.modules())

    @pytest.mark.parametrize("factory", _WEIGHTED)
    def test_weighted_factories_are_eval(self, factory, monkeypatch):
        # SOLD2_detector fetches the wireframe checkpoint from a URL whose file CI does not prefetch; SOLD2's mirror
        # serves the same bytes and is prefetched.
        monkeypatch.setitem(sold2_detector.urls, "wireframe", sold2.urls["wireframe"])
        model = factory()
        assert not model.training
        assert not any(m.training for m in model.modules())

    def test_non_default_architecture_needs_pretrained_false(self):
        # The liberty checkpoint has convolution biases, so it cannot load strictly into bias-free convolutions.
        with pytest.raises(RuntimeError):
            HyNet(is_bias=False)
        model = HyNet(pretrained=False, is_bias=False)
        assert model.layer2[0].bias is None

    def test_lightglue_mode(self, monkeypatch):
        assert LightGlue(features=None).training
        # "superpoint" keeps the default architecture, so a fresh default model provides a complete state dict.
        weights = LightGlue(features=None).state_dict()
        monkeypatch.setattr("kornia.feature.lightglue.load_state_dict_from_url", lambda url, **kwargs: dict(weights))
        model = LightGlue("superpoint")
        assert not model.training
        assert not any(m.training for m in model.modules())
        # weights count as loaded only when the state dict holds any
        monkeypatch.setattr("kornia.feature.lightglue.load_state_dict_from_url", lambda url, **kwargs: {})
        assert LightGlue("superpoint").training

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("source", ["url", "file"])
    @pytest.mark.parametrize(
        "checkpoint_kind", ["complete", "missing_buffer", "missing_parameter", "buffers_only", "empty"]
    )
    def test_lightglue_checkpoint_completeness_controls_mode(self, monkeypatch, tmp_path, source, checkpoint_kind):
        config = {"input_dim": 256, "descriptor_dim": 32, "n_layers": 2, "flash": False}
        weights = LightGlue(features=None, **config).state_dict()
        if checkpoint_kind == "missing_buffer":
            # Reference checkpoints omit this derived buffer, but still contain every learned parameter.
            del weights["confidence_thresholds"]
        elif checkpoint_kind == "missing_parameter":
            del weights["posenc.Wr.weight"]
        elif checkpoint_kind == "buffers_only":
            weights = {"confidence_thresholds": weights["confidence_thresholds"]}
        elif checkpoint_kind == "empty":
            weights = {}

        if source == "url":
            monkeypatch.setattr(
                "kornia.feature.lightglue.load_state_dict_from_url", lambda *args, **kwargs: dict(weights)
            )
            model = LightGlue("superpoint", **config)
        else:
            # Local checkpoints are resolved relative to lightglue.py's weights directory.
            monkeypatch.setattr("kornia.feature.lightglue.__file__", str(tmp_path / "lightglue.py"))
            (tmp_path / "weights").mkdir()
            torch.save(weights, tmp_path / "weights" / "checkpoint.pth")
            model = LightGlue(features=None, weights="checkpoint", **config)

        expected_training = checkpoint_kind not in ("complete", "missing_buffer")
        assert {module.training for module in model.modules()} == {expected_training}
        for name, value in weights.items():
            self.assert_close(model.state_dict()[name], value)

    @pytest.mark.slow
    def test_composites_build_pretrained_eval_modules(self):
        # The containers (LocalFeature, KeyNetDetector, LAFOrienter, LAFDescriptor) never set their own mode; the
        # learned networks they hold are built with their weights and come back in evaluation mode.
        learned = (HardNet, OriNet, LAFAffNetShapeEstimator, KeyNet)
        composite = KeyNetAffNetHardNet(num_features=8)
        networks = [m for m in composite.modules() if isinstance(m, learned)]
        assert {type(m) for m in networks} == set(learned)
        assert not any(m.training for net in networks for m in net.modules())

        descriptor = LAFDescriptor(None)
        networks = [m for m in descriptor.modules() if isinstance(m, learned)]
        assert [type(m) for m in networks] == [HardNet]
        assert not any(m.training for m in networks[0].modules())
