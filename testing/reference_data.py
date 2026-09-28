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

URLs are commit-pinned ``raw.githubusercontent.com`` links, which avoid the
GitHub blob-page redirect the ``?raw=true`` spelling goes through.
"""

from __future__ import annotations

from kornia.filters import dexined as _dexined

# Test data commit hashes from the kornia/data_test repository.
DATA_TEST_SHA: dict[str, str] = {
    "loftr": "cb8f42bf28b9f347df6afba5558738f62a11f28a",
    "adalam": "f7d8da661701424babb64850e03c5e8faec7ea62",
    "disk": "8b98f44abbe92b7a84631ed06613b08fee7dae14",
    "xfeat": "279e95e411f2d3926953dea3842347242190f4da",
}

_RAW = "https://raw.githubusercontent.com/kornia/data_test"

TEST_DATA_URLS: dict[str, str | list[str]] = {
    "loftr_homo": f"{_RAW}/{DATA_TEST_SHA['loftr']}/loftr_outdoor_and_homography_data.pt",
    "loftr_fund": f"{_RAW}/{DATA_TEST_SHA['loftr']}/loftr_indoor_and_fundamental_data.pt",
    "adalam_idxs": f"{_RAW}/{DATA_TEST_SHA['adalam']}/adalam_test.pt",
    "lightglue_idxs": f"{_RAW}/{DATA_TEST_SHA['adalam']}/adalam_test.pt",
    "disk_outdoor": f"{_RAW}/{DATA_TEST_SHA['disk']}/knchurch_disk.pt",
    "xfeat_outdoor": f"{_RAW}/{DATA_TEST_SHA['xfeat']}/xfeat_reference.pt",
    # The library's own DexiNed checkpoint, from the library's own source list, so the
    # fixture shares its cache entry, its fallback mirror and its prefetch guard.
    "dexined": list(_dexined.url),
}

__all__ = ["DATA_TEST_SHA", "TEST_DATA_URLS"]
