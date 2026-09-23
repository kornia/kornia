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

import inspect

import pytest

import kornia.geometry.transform as T

# Signature defaults that the Convention blocks or the conventions page state because they differ
# from a sibling (the 2D matrix warps and crops default to align_corners=True). One row per
# stated default.
_STATED_DEFAULTS = [
    ("warp_perspective", "align_corners", True),
    ("warp_affine", "align_corners", True),
    ("remap", "align_corners", None),
    ("elastic_transform2d", "align_corners", False),
    ("warp_image_tps", "align_corners", False),
    ("crop_by_indices", "align_corners", None),
    ("affine3d", "align_corners", False),
    ("rotate3d", "align_corners", False),
    ("shear", "align_corners", False),
    ("Shear", "align_corners", True),
    ("rescale", "align_corners", None),
    ("Rescale", "align_corners", True),
]


@pytest.mark.parametrize(("name", "param", "default"), _STATED_DEFAULTS, ids=str)
def test_convention_stated_signature_default(name, param, default):
    obj = getattr(T, name)
    sig = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj)
    assert sig.parameters[param].default == default
    assert type(sig.parameters[param].default) is type(default)
