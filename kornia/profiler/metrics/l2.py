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


def l2_normalized(f1, f2):
    """Normalized L2 distance per layer."""
    if f1.keys() != f2.keys():
        raise ValueError("Feature dictionaries must have same layers")

    results = {}

    for layer in f1:
        x = f1[layer].reshape(f1[layer].size(0), -1)
        y = f2[layer].reshape(f2[layer].size(0), -1)

        l2 = torch.norm(x - y, p=2, dim=1).mean()
        norm = torch.norm(x, p=2, dim=1).mean()

        results[layer] = (l2 / (norm + 1e-8)).item()

    return results
