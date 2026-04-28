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

"""kornia.augmentations — canonical namespace.

Transparent alias of kornia.augmentation. All existing import paths under
kornia.augmentation continue to work. New code should prefer the plural form.
"""

from __future__ import annotations

import importlib
import sys

from kornia.augmentation import *
from kornia.augmentation import __all__

_SINGULAR = "kornia.augmentation"


def __getattr__(name: str) -> object:
    """Lazy resolution for sub-modules.

    Maps kornia.augmentations.<name> -> kornia.augmentation.<name>.
    """
    try:
        mod = importlib.import_module(f"{_SINGULAR}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(name) from exc
    # Register so future lookups skip __getattr__
    sys.modules[f"{__name__}.{name}"] = mod
    return mod


# Mirror every already-imported kornia.augmentation.* sub-module under the plural
# namespace so deep paths like `from kornia.augmentations._2d.geometric.X import Y`
# resolve to the same module objects as the singular path.
_prefix = f"{_SINGULAR}."
_plural_prefix = f"{__name__}."
for _key, _mod in list(sys.modules.items()):
    if _key.startswith(_prefix):
        _suffix = _key[len(_prefix) :]
        sys.modules[f"{_plural_prefix}{_suffix}"] = _mod
