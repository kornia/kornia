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

import sys

import pytest

from kornia.utils._compat import torch_version

# NOTE: remove once we deprecate pytorch 1.10.2 for macos

if sys.platform == "darwin" and torch_version() == "1.10.2":
    pytest.skip("Accelerate is broken for macos and pytorch 1.10.2", allow_module_level=True)
