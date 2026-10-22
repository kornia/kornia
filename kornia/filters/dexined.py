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

"""DexiNed under its old import path.

The network lives in :mod:`kornia.models.dexined`. This module re-exports it and its building blocks, so
``kornia.filters.dexined.DexiNed`` is the same class as :class:`kornia.models.dexined.DexiNed`. The private blocks
are re-exported too: a whole model pickled with ``torch.save`` from this module names them.
"""

from kornia.models.dexined import (  # noqa: F401
    CoFusion,
    DexiNed,
    DoubleConvBlock,
    SingleConvBlock,
    UpConvBlock,
    _DenseBlock,
    _DenseLayer,
    weight_init,
)
from kornia.models.dexined import _url as url  # noqa: F401
