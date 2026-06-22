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

import subprocess
import sys
import textwrap


def test_import_without_jit_script_deprecation():
    """Importing kornia must not eagerly call ``torch.jit.script``.

    ``torch.jit.script`` is deprecated (and unsupported on Python 3.14+). Decorating
    functions with ``@torch.jit.script`` runs the deprecated path at import time, which
    makes ``python -W error -c "import kornia"`` fail. This guards against reintroducing
    an import-time scripting call. See https://github.com/kornia/kornia/issues/3727.

    A fresh interpreter is used so the result is independent of kornia already being
    imported by the test session. Only the ``torch.jit.script`` deprecation is promoted
    to an error to avoid coupling the test to unrelated third-party warnings.
    """
    script = textwrap.dedent(
        """
        import warnings

        warnings.filterwarnings("error", message=r".*torch\\.jit\\.script.*")
        warnings.filterwarnings("error", category=DeprecationWarning, module=r"kornia.*")
        warnings.filterwarnings("error", category=FutureWarning, module=r"kornia.*")

        import kornia  # noqa: F401
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"importing kornia raised a deprecation warning:\n{result.stderr}"
