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
import re

import pytest

import kornia.augmentation as K


def _documented_args(cls):
    doc = inspect.getdoc(cls) or ""
    block = re.search(r"Args:\n(.*?)\n\s*(?:Shape|Returns|Examples?|\.\. note::|Note)", doc, re.S)
    assert block, f"{cls.__name__} has no Args block"
    return [match.group(1) for match in re.finditer(r"^\s{4}(\w+):", block.group(1), re.M)]


@pytest.mark.parametrize("cls", [K.ColorJitter, K.ColorJiggle, K.RandomBrightness, K.RandomGaussianBlur])
def test_args_block_matches_the_constructor(cls):
    # kornia#4437: three classes documented a `silence_instantiation_warning` argument that does not
    # exist, and ColorJitter did not document its real `order` argument.
    signature = list(inspect.signature(cls.__init__).parameters)[1:]

    assert sorted(_documented_args(cls)) == sorted(signature)
