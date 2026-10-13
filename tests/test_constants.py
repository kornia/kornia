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

import re

import pytest
import torch

import kornia.augmentation as K
from kornia.constants import BorderType, DataKey, DType, Resample, SamplePadding

from testing.base import BaseTester

_ENUMS = (Resample, BorderType, SamplePadding, DType, DataKey)


class TestEnumGet(BaseTester):
    @pytest.mark.parametrize("enum", _ENUMS)
    def test_names_values_and_members_resolve(self, enum):
        for name, member in enum.__members__.items():
            assert enum.get(name) is member
            assert enum.get(name.lower()) is member
            assert enum.get(member) is member
            assert enum.get(member.value) is member

    @pytest.mark.parametrize("enum", _ENUMS)
    def test_unknown_name_is_both_a_key_error_and_a_value_error_5907(self, enum):
        with pytest.raises(ValueError) as excinfo:
            enum.get("bogus")
        assert isinstance(excinfo.value, KeyError)

    @pytest.mark.parametrize("enum", _ENUMS)
    def test_unknown_name_message_names_the_value_and_the_valid_names_5907(self, enum):
        with pytest.raises(KeyError) as excinfo:
            enum.get("Bogus")
        message = str(excinfo.value)
        # the value as given, and no repr quoting of the whole message as a plain KeyError's str() would add
        assert message.startswith(f"'Bogus' is not a valid {enum.__name__}")
        for name in enum.__members__:
            assert repr(name.lower()) in message

    def test_out_of_range_int_is_a_value_error(self):
        with pytest.raises(ValueError, match="7 is not a valid Resample"):
            Resample.get(7)

    @pytest.mark.parametrize("enum", _ENUMS)
    @pytest.mark.parametrize("value", [BorderType.REFLECT, Resample.BILINEAR, 1.5, None])
    def test_wrong_type_message_names_the_enum_and_the_value_5907(self, enum, value):
        if isinstance(value, enum):
            pytest.skip("a member of the enum itself is accepted")
        with pytest.raises(TypeError) as excinfo:
            enum.get(value)
        message = str(excinfo.value)
        assert enum.__name__ in message
        assert repr(value) in message
        # `_KORNIA_EnumMeta.__repr__` joins every member of an enum with " | "
        assert " | " not in message

    def test_dtype_rejects_an_unsupported_torch_dtype_5907(self):
        with pytest.raises(ValueError) as excinfo:
            DType.get(torch.uint8)
        assert isinstance(excinfo.value, KeyError)
        message = str(excinfo.value)
        assert message.startswith("torch.uint8 is not a valid DType")
        assert "'float32'" in message

    def test_dynamo_name_lookup_fullgraph_5907(self, device, dtype, torch_optimizer):
        # augmentations look names up in their forward too (a per-call `border_type`, `data_keys`)
        def op(x):
            return x * (BorderType.get("reflect").value + Resample.get("Bilinear").value + DType.get(x.dtype).value)

        x = torch.ones(2, device=device, dtype=dtype)
        self.assert_close(torch_optimizer(op, fullgraph=True)(x), op(x))


class TestMisspeltNameThroughThePublicAPI(BaseTester):
    @pytest.mark.parametrize(
        "make,match",
        [
            pytest.param(
                lambda: K.RandomAffine(10.0, resample="bilinera"),
                "'bilinera' is not a valid Resample",
                id="RandomAffine-resample",
            ),
            pytest.param(
                lambda: K.RandomAffine(10.0, padding_mode="zero"),
                "'zero' is not a valid SamplePadding",
                id="RandomAffine-padding_mode",
            ),
            pytest.param(
                lambda: K.CenterCrop(4, resample="bicubik"),
                "'bicubik' is not a valid Resample",
                id="CenterCrop-resample",
            ),
            pytest.param(
                lambda: K.RandomRotation3D(10.0, resample="nearst"),
                "'nearst' is not a valid Resample",
                id="RandomRotation3D-resample",
            ),
            pytest.param(
                lambda: K.RandomGaussianBlur((3, 3), (0.1, 2.0), border_type="reflct"),
                "'reflct' is not a valid BorderType",
                id="RandomGaussianBlur-border_type",
            ),
            pytest.param(
                lambda: K.RandomMotionBlur(3, 35.0, 0.5, border_type="replcate"),
                "'replcate' is not a valid BorderType",
                id="RandomMotionBlur-border_type",
            ),
            pytest.param(
                lambda: K.AugmentationSequential(K.ColorJiggle(0.1, p=1.0), data_keys=["input", "bbox_yxyx"]),
                "'bbox_yxyx' is not a valid DataKey",
                id="AugmentationSequential-data_keys",
            ),
        ],
    )
    def test_constructor_raises_a_value_error_that_except_key_error_still_catches_5907(self, make, match):
        with pytest.raises(ValueError, match=re.escape(match)) as excinfo:
            make()
        assert isinstance(excinfo.value, KeyError)

    def test_misspelt_per_call_border_type_override_5907(self, device, dtype):
        img = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        aug = K.RandomBoxBlur((3, 3), p=1.0)
        with pytest.raises(ValueError, match=re.escape("'reflct' is not a valid BorderType")):
            aug(img, border_type="reflct")

    def test_valid_names_are_listed_5907(self):
        with pytest.raises(ValueError) as excinfo:
            K.RandomAffine(10.0, padding_mode="zero")
        message = str(excinfo.value)
        for name in ("'zeros'", "'border'", "'reflection'", "'fill'"):
            assert name in message
