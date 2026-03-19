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

import torch

import kornia as K
from docs.generate_examples import apply_augmentation, handle_special_cases


def test_handle_special_cases_jigsaw():
    img = torch.randn(1, 3, 100, 100)
    out = handle_special_cases("RandomJigsaw", img)
    assert out.shape[-2:] == (1020, 500)


def test_handle_special_cases_jpeg():
    img = torch.randn(1, 3, 200, 200)
    out = handle_special_cases("RandomJPEG", img)
    assert out.shape[-2] == 176


def test_apply_augmentation_runs():
    img = torch.randn(2, 3, 64, 64)
    out = apply_augmentation(K.augmentation, "RandomHorizontalFlip", (), 42, img)
    assert out.shape == img.shape


if __name__ == "__main__":
    test_handle_special_cases_jigsaw()
    test_handle_special_cases_jpeg()
    test_apply_augmentation_runs()
    print("All tests passed!")
