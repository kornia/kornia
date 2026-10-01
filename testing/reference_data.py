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


"""Reference tensors the ``data`` fixture in ``conftest.py`` loads.

The table lives here rather than in ``conftest.py`` so that
``tests/core/test_weights_prefetch.py`` can import it as an ordinary module and
enumerate it in ``WEIGHT_REGISTRIES`` next to the library's own weight tables:
every entry must be prefetched by ``.github/download-models-weights.py`` under
the exact cache name the fixture looks up, from the exact source list.

The reference tensors are ``.safetensors`` files in ``kornia/data_test``, read
without unpickling anything. Their URLs are commit-pinned
``raw.githubusercontent.com`` links, which avoid the GitHub blob-page redirect
the ``?raw=true`` spelling goes through.
"""

from __future__ import annotations

from typing import Any

from kornia.core import check_safetensors, download_file_from_url, load_safetensors
from kornia.feature import DISKFeatures
from kornia.filters import dexined as _dexined

# The kornia/data_test commit that carries every ``.safetensors`` reference file.
DATA_TEST_SHA = "4ffed08df3d82af85aa9012d3104f19ca4b62604"

_RAW = f"https://raw.githubusercontent.com/kornia/data_test/{DATA_TEST_SHA}"

# Reference tensors, read by :func:`load_reference_data`.
TEST_DATA_URLS: dict[str, str | list[str]] = {
    "loftr_homo": f"{_RAW}/loftr_outdoor_and_homography_data.safetensors",
    "loftr_fund": f"{_RAW}/loftr_indoor_and_fundamental_data.safetensors",
    "adalam_idxs": f"{_RAW}/adalam_test.safetensors",
    "lightglue_idxs": f"{_RAW}/adalam_test.safetensors",
    "disk_outdoor": f"{_RAW}/knchurch_disk.safetensors",
    "xfeat_outdoor": f"{_RAW}/xfeat_reference.safetensors",
}

# Library checkpoints the fixture serves as they are: the library's own source list,
# loaded the library's way, so the fixture shares its cache entry, its fallback
# mirror and its prefetch guard.
TEST_CHECKPOINT_URLS: dict[str, str | list[str]] = {
    "dexined": list(_dexined.url),
}

# Entries whose file stores ``list[DISKFeatures]`` values of length 1, flattened to
# ``<key>.keypoints``, ``<key>.descriptors`` and ``<key>.detection_scores``.
_DISK_FEATURES: dict[str, tuple[str, ...]] = {
    "disk_outdoor": ("disk1", "disk2"),
}


def load_reference_data(name: str) -> dict[str, Any]:
    """Download and read the reference tensors of the ``TEST_DATA_URLS`` entry *name*.

    A truncated cache entry fails ``check_safetensors`` and is fetched again
    rather than read.

    Returns:
        The file's tensors by key, on CPU, with each flattened ``DISKFeatures``
        rebuilt into the one-element list the tests index.
    """
    path = download_file_from_url(TEST_DATA_URLS[name], validate=check_safetensors)
    data: dict[str, Any] = load_safetensors(path)
    for key in _DISK_FEATURES.get(name, ()):
        fields = ("keypoints", "descriptors", "detection_scores")
        data[key] = [DISKFeatures(*(data.pop(f"{key}.{field}") for field in fields))]
    return data


__all__ = ["DATA_TEST_SHA", "TEST_CHECKPOINT_URLS", "TEST_DATA_URLS", "load_reference_data"]
