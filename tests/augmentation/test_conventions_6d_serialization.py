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

import copy
import pickle

import pytest
import torch

import kornia.augmentation as K
import kornia.augmentation.auto as A

from testing.base import BaseTester

_CASES = [
    ("RandomAffine3D", {"degrees": 15.0}),
    ("CenterCrop3D", {"size": (3, 4, 5)}),
    ("RandomCrop3D", {"size": (3, 4, 5)}),
    ("RandomDepthicalFlip3D", {}),
    ("RandomHorizontalFlip3D", {}),
    ("RandomPerspective3D", {"distortion_scale": 0.2}),
    ("RandomRotation3D", {"degrees": 15.0}),
    ("RandomVerticalFlip3D", {}),
    ("RandomEqualize3D", {}),
    ("RandomMotionBlur3D", {"kernel_size": 3, "angle": 15.0, "direction": 0.3}),
    ("RandomCutMixV2", {"use_correct_lambda": True}),
    ("RandomJigsaw", {}),
    ("RandomMixUpV2", {}),
    ("RandomMosaic", {}),
    ("PatchMix", {"patch_size": 2}),
]


@pytest.mark.device_agnostic
class TestConventionMix3DSerialization(BaseTester):
    @pytest.mark.parametrize("name,kwargs", _CASES, ids=[name for name, _ in _CASES])
    @pytest.mark.parametrize("batched,keepdim", [(True, False), (False, False), (False, True)])
    def test_convention_numeric_config_pickle_and_deepcopy_preserve_replay(self, name, kwargs, batched, keepdim):
        # Numeric configurations have no state_dict entries; pickle/deepcopy also preserve
        # configuration and last parameters. A state_dict is therefore not a replay checkpoint.
        torch.manual_seed(17)
        augmentation = getattr(K, name)(p=1.0, keepdim=keepdim, **kwargs)
        image = torch.rand((2, 3, 6, 8, 10) if name.endswith("3D") else (2, 3, 8, 12))
        if not batched:
            image = image[0]
        batch_shape = image.shape if batched else torch.Size((1, *image.shape))
        expected = augmentation(image)
        assert expected.ndim == image.ndim + int(not batched and not keepdim)
        assert not augmentation.state_dict()
        assert not list(augmentation.parameters())
        assert not list(augmentation.buffers())
        for restored in (pickle.loads(pickle.dumps(augmentation)), copy.deepcopy(augmentation)):  # noqa: S301
            assert restored._params is not augmentation._params
            assert restored._params.keys() == augmentation._params.keys()
            for key in augmentation._params:
                self.assert_close(restored._params[key], augmentation._params[key], rtol=0, atol=0)
            self.assert_close(restored(image, params=restored._params), expected, rtol=0, atol=0)
            # Fresh draws consult constructor ranges/samplers rather than the replay dictionary.
            torch.manual_seed(31)
            reference_params = augmentation.forward_parameters(batch_shape)
            torch.manual_seed(31)
            restored_params = restored.forward_parameters(batch_shape)
            assert restored_params.keys() == reference_params.keys()
            for key in reference_params:
                self.assert_close(restored_params[key], reference_params[key], rtol=0, atol=0)

    @pytest.mark.parametrize(
        "name,shape", [("RandomTransplantation", (3, 2, 6, 8)), ("RandomTransplantation3D", (3, 2, 4, 6, 8))]
    )
    def test_convention_transplantation_pickle_and_deepcopy_preserve_replay(self, name, shape):
        # The serialization bullet names both transplantation classes; they take a mask, so they need their own case.
        torch.manual_seed(17)
        augmentation = getattr(K, name)(p=1.0)
        image = torch.rand(shape)
        mask = torch.randint(0, 3, (shape[0], *shape[2:]))
        expected, _ = augmentation(image, mask, data_keys=["input", "mask"])
        assert not torch.equal(expected, image)
        assert not augmentation.state_dict()
        for restored in (pickle.loads(pickle.dumps(augmentation)), copy.deepcopy(augmentation)):  # noqa: S301
            assert restored._params is not augmentation._params
            assert restored._params.keys() == augmentation._params.keys()
            for key in augmentation._params:
                self.assert_close(restored._params[key], augmentation._params[key], rtol=0, atol=0)
            replayed, _ = restored(image, mask, params=restored._params, data_keys=["input", "mask"])
            self.assert_close(replayed, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("name", ["AutoAugment", "RandAugment", "TrivialAugment"])
    def test_wart_auto_policies_cannot_be_pickled_4469(self, name):
        kwargs = {"n": 2, "m": 15} if name == "RandAugment" else {}
        augmentation = getattr(A, name)(**kwargs)
        with pytest.raises(AttributeError, match="local object"):
            pickle.dumps(augmentation)

    @pytest.mark.parametrize("name", ["AutoAugment", "RandAugment", "TrivialAugment"])
    def test_convention_a_posterize_only_policy_can_be_pickled(self, name):
        # A policy pickles when every wrapper in it does; Posterize passes a named mapping and no sign flip.
        entry = ("posterize", 1.0, 3) if name == "AutoAugment" else ("posterize", 0.0, 4)
        kwargs = {"n": 1, "m": 15} if name == "RandAugment" else {}
        augmentation = getattr(A, name)(policy=[[entry]], **kwargs)
        image = torch.rand(2, 3, 4, 4)
        expected = augmentation(image)
        restored = pickle.loads(pickle.dumps(augmentation))  # noqa: S301
        self.assert_close(restored(image, params=restored._params), expected, rtol=0, atol=0)

    def test_convention_empty_autoaugment_policy_can_be_pickled(self):
        # With no operation wrappers, there is no local magnitude function blocking pickle.
        augmentation = A.AutoAugment(policy=[[]])
        image = torch.rand(2, 1, 4, 4)
        augmentation(image)
        restored = pickle.loads(pickle.dumps(augmentation))  # noqa: S301
        self.assert_close(restored(image, params=restored._params), image, rtol=0, atol=0)
        assert restored.transform_matrix is None
