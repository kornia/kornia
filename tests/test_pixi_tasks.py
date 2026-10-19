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

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIXI = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
pytestmark = pytest.mark.device_agnostic


@pytest.mark.parametrize("task", ["install", "install-docs"])
def test_cuda_index_is_a_url(task: str) -> None:
    args = shlex.split(PIXI["feature"]["cuda"]["tasks"][task]["cmd"])
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cuda_index = next(index for index in project["tool"]["uv"]["index"] if index["name"] == "pytorch-cu121")
    # --index takes a URL, not the configured index's name.
    assert args[args.index("--index") + 1] == cuda_index["url"]


@pytest.mark.parametrize("task", ["install", "install-docs"])
@pytest.mark.parametrize("activation", ["CONDA_PREFIX", "VIRTUAL_ENV"])
def test_cuda_install_and_run_share_environment(tmp_path: Path, task: str, activation: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required; run this test through Pixi")
    env = {key: value for key, value in os.environ.items() if not key.startswith("UV_")}
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"):
        env.pop(key, None)
    cuda = PIXI["feature"]["cuda"]
    # Activation of every feature the cuda environment composes, as Pixi applies it.
    env.update(PIXI.get("activation", {}).get("env", {}))
    for feature in PIXI["environments"]["cuda"]["features"]:
        env.update(PIXI["feature"][feature].get("activation", {}).get("env", {}))
    project_env = env.get("UV_PROJECT_ENVIRONMENT", ".venv")
    env.update(UV_CACHE_DIR=str(tmp_path / "cache"), UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never")
    env["PATH"] = str(Path(uv).parent) + os.pathsep + env["PATH"]

    def run(args: list[str]) -> str:
        result = subprocess.run(  # noqa: S603
            args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    # Tiny metadata-only wheels exercise real uv selection and re-sync behavior,
    # without a GPU, network access, or a PyTorch download.
    for version in ("2.14.0", "2.5.1+cu121"):
        info = f"torch-{version}.dist-info"
        files = {
            f"{info}/METADATA": f"Metadata-Version: 2.1\nName: torch\nVersion: {version}\n",
            f"{info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        }
        files[f"{info}/RECORD"] = "".join(f"{name},,\n" for name in [*files, f"{info}/RECORD"])
        with ZipFile(tmp_path / f"torch-{version}-py3-none-any.whl", "w") as wheel:
            for name, content in files.items():
                wheel.writestr(name, content)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "pixi-task-probe"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["torch"]\n[project.optional-dependencies]\ndev = []\ndocs = []\n'
        '[tool.uv.sources]\ntorch = {path = "torch-2.14.0-py3-none-any.whl"}\n',
        encoding="utf-8",
    )
    index = tmp_path / "simple" / "torch"
    index.mkdir(parents=True)
    (index / "index.html").write_text(
        '<a href="../../torch-2.5.1%2Bcu121-py3-none-any.whl">torch</a>', encoding="utf-8"
    )
    pixi_env = tmp_path / ".pixi" / "envs" / "cuda"
    run([uv, "venv", "--python", sys.executable, str(pixi_env)])
    # uv targets CONDA_PREFIX only when it is not a base environment; this marker is how it recognizes a Pixi
    # environment, whatever CONDA_DEFAULT_ENV says. Without it the CONDA_PREFIX case passes on the old command.
    (pixi_env / "conda-meta").mkdir()
    (pixi_env / "conda-meta" / "pixi").write_text("{}", encoding="utf-8")
    run([uv, "venv", "--python", sys.executable, project_env])
    env[activation] = str(pixi_env)
    if activation == "CONDA_PREFIX":
        env["CONDA_DEFAULT_ENV"] = "kornia:cuda"  # as `pixi run -e cuda` sets it
    for command in cuda["tasks"][task]["cmd"].split(" && "):
        args = shlex.split(command)
        if "--index" in args:
            args[args.index("--index") + 1] = index.parent.as_uri()
        run([uv, *args[1:]])

    # Replace only pytest with a metadata probe; preserve every uv run option
    # from the CUDA test tasks and the inherited focused-test task.
    probe = "import json,sys; from importlib.metadata import version; print(json.dumps([sys.prefix,version('torch')]))"
    tasks = PIXI["tasks"] | cuda["tasks"]
    for name in ("test-cuda", "test-cuda-f32", "test-cuda-f64", "test-cuda-half", "test-module"):
        args = shlex.split(tasks[name]["cmd"])
        env.update(tasks[name].get("env", {}))
        prefix, version = json.loads(run([uv, *args[1 : args.index("pytest")], "python", "-c", probe]))
        assert Path(prefix) == tmp_path / project_env
        assert version == "2.5.1+cu121"
    assert json.loads(run([uv, "pip", "list", "--python", str(pixi_env), "--format", "json"])) == []


@pytest.mark.parametrize("environment", ["default", "py311", "py312", "py313", "py314"])
def test_non_cuda_environment_selection(environment: str) -> None:
    env = dict(PIXI.get("activation", {}).get("env", {}))
    for feature in PIXI["environments"][environment]["features"]:
        env.update(PIXI["feature"][feature].get("activation", {}).get("env", {}))
    assert "UV_NO_SYNC" not in env
    expected = ".venv" if environment in ("default", "py311") else f".venv-{environment}"
    assert env.get("UV_PROJECT_ENVIRONMENT", ".venv") == expected
