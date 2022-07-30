import sys

import pytest

from kornia.utils._compat import torch_version

if sys.platform == "Darwin" and torch_version() == "1.10.2":
    pytest.skip("Accelerate is broken for macos and pytorch 1.10.2", allow_module_level=True)
