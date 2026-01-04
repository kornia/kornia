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

import torch
from torch.utils import benchmark

# Compare takes a list of measurements which we'll save in results.
results = []

sizes = [1, 64, 102]
for b, _ in product(sizes, sizes):
    # label and sub_label are the rows
    # description is the column
    label = "get_perspective_transform"
    sub_label = f"[{b}, {4}, {2}]"
    x = torch.rand((b, 4, 2))
    for num_threads in [1, 4, 16, 32]:
        results.append(
            benchmark.Timer(
                stmt="get_perspective_transform(x, x)",
                setup="from kornia.geometry import get_perspective_transform",
                globals={"x": x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description="get_perspective_transform",
            ).blocked_autorange(min_run_time=1)
        )

compare = benchmark.Compare(results)
compare.print()
