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

from itertools import product

import pytest


@pytest.fixture
def shape(B, C, H, W):
    return (B, C, H, W)


def pytest_generate_tests(metafunc):
    B = [1, 5]
    C = [1, 3]
    H = W = [128]  # , 237, 512]
    params = list(product(B, C, H, W))
    metafunc.parametrize("B, C, H, W", params)
