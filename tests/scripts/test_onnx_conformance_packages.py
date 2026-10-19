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

import re
import subprocess
import sys
from pathlib import Path

import pytest

from testing import onnx_conformance

# `.github/scripts/` is neither a package nor on the path; see tests/scripts/test_check_torch_env.py for why this
# stays a per-module insert.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".github" / "scripts"))

from onnx_conformance_packages import STABLE_CORE, select

REPO = Path(__file__).parent.parent.parent
SCRIPT = REPO / ".github" / "scripts" / "onnx_conformance_packages.py"

pytestmark = pytest.mark.device_agnostic


def _selected(paths):
    result = select(paths)
    return set(result.split(",")) if result else set()


@pytest.mark.parametrize(
    "changed",
    [
        ["kornia/core/check.py"],
        ["kornia/utils/helpers.py"],
        ["kornia/onnx/sequential.py"],
        ["kornia/__init__.py"],
        ["kornia/constants.py"],
        ["docs/export_support/cases_misc.py"],
        ["docs/source/_data/export_support.json"],
        ["docs/source/get-started/stability.rst"],
        ["testing/known_failure_xfails/onnx_conformance_torch2.14.txt"],
        ["tests/onnx/test_conformance.py"],
        [".github/workflows/pr_test_cpu.yml"],
        ["uv.lock"],
        ["pixi.toml"],
        ["pyproject.toml"],
        ["conftest.py"],
    ],
)
def test_select_all(changed):
    assert select(changed) == "all"


@pytest.mark.parametrize(
    "changed",
    [
        ["docs/source/get-started/onnx.rst", "README.md"],
        ["tests/geometry/test_homography.py"],
        ["kornia/contrib/face_detection.py", "kornia/models/sam/model.py"],
        ["kornia/x/trainer.py"],
        [],
    ],
)
def test_select_nothing(changed):
    assert select(changed) == ""


def test_select_follows_importers_in_the_repository():
    # The PSNR losses call kornia.metrics.psnr: a fix there must run the losses' known failures too (#6165).
    assert _selected(["kornia/metrics/psnr.py"]) >= {"metrics", "losses"}
    assert "metrics" not in _selected(["kornia/losses/psnr.py"])
    assert _selected(["kornia/geometry/transform/imgwarp.py"]) >= {"geometry", "augmentation", "feature"}
    # PatchSequential crops with kornia.contrib.extract_patches; the DexiNed filter re-exports kornia.models.dexined.
    assert _selected(["kornia/contrib/extract_patches.py"]) >= {"augmentation"}
    assert _selected(["kornia/models/dexined.py"]) >= {"filters"}


def _write(root, files):
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def test_select_resolves_each_import_form(tmp_path):
    _write(
        tmp_path,
        {
            "kornia/__init__.py": "from . import augmentation, color, contrib, feature, filters, geometry, losses\n",
            "kornia/metrics/__init__.py": "from .psnr import psnr\n",
            "kornia/metrics/psnr.py": "def psnr(): ...\n",
            "kornia/losses/__init__.py": "",
            "kornia/losses/psnr.py": "from kornia import metrics\n",  # a submodule imported from its package
            "kornia/color/__init__.py": "",
            "kornia/color/hsv.py": "def rgb_to_hsv(): ...\n",
            "kornia/augmentation/__init__.py": "",
            "kornia/augmentation/aug.py": "import kornia\n\nf = kornia.color.hsv.rgb_to_hsv\n",  # attribute chain
            "kornia/geometry/__init__.py": "",
            "kornia/geometry/y.py": "def g(): ...\n",
            "kornia/feature/__init__.py": "",
            "kornia/feature/x.py": "def f():\n    from ..geometry.y import g\n",  # relative, function-local
            "kornia/contrib/__init__.py": "",
            "kornia/contrib/c.py": "",
            "kornia/models/__init__.py": "",
            "kornia/models/dex.py": "class N: ...\n",
            "kornia/models/sam.py": "",
            "kornia/filters/__init__.py": "",
            "kornia/filters/d.py": "from kornia.models.dex import N\n",  # a name from a module
        },
    )
    assert select(["kornia/metrics/psnr.py"], tmp_path) == "losses,metrics"
    assert select(["kornia/color/hsv.py"], tmp_path) == "augmentation,color"
    assert select(["kornia/geometry/y.py"], tmp_path) == "geometry,feature"
    assert select(["kornia/models/dex.py"], tmp_path) == "filters"
    assert select(["kornia/contrib/c.py", "kornia/models/sam.py"], tmp_path) == ""
    assert select(["kornia/color/removed.py"], tmp_path) == "color"  # a deleted module stands for its package


@pytest.mark.parametrize(
    "stdin",
    [
        "kornia/morphology/morphology.py\nkornia/x/trainer.py\n",  # git diff --name-only
        "kornia/morphology/morphology.py kornia/x/trainer.py\n",  # echo a b
    ],
)
def test_cli_reads_stdin(stdin):
    result = subprocess.run(  # noqa: S603 -- the current test interpreter on the script under test.
        [sys.executable, str(SCRIPT)],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "morphology\n"


def test_cli_reads_whitespace_separated_paths():
    # One echo line with several paths must select what the paths select one per line.
    result = subprocess.run(  # noqa: S603 -- the current test interpreter on the script under test.
        [sys.executable, str(SCRIPT)],
        input="kornia/morphology/morphology.py kornia/core/check.py\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "all\n"


def test_stable_core_matches_the_test_and_the_stability_page():
    assert STABLE_CORE == onnx_conformance.STABLE_CORE
    assert set(onnx_conformance.REGISTRIES_OF) == set(STABLE_CORE)
    page = (REPO / "docs" / "source" / "get-started" / "stability.rst").read_text(encoding="utf-8")
    paragraph = page.split("**Stable core**", 1)[1].split("\n\n", 1)[0]
    assert set(re.findall(r"``kornia\.(\w+)``", paragraph)) == set(STABLE_CORE)
