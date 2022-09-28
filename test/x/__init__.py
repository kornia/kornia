from __future__ import annotations

import sys

import pytest

from kornia.utils._compat import torch_version

# NOTE: remove once we deprecate pytorch 1.10.2 for macos

if sys.platform == "darwin" and torch_version() == "1.10.2":
    pytest.skip("Accelerate is broken for macos and pytorch 1.10.2", allow_module_level=True)
