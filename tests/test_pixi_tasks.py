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
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIXI = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
LOCKED_TORCH = next(
    package
    for package in tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    if package["name"] == "torch"
)
WINDOWS_CUDA_BUILD = "cu132"
pytestmark = pytest.mark.device_agnostic


def _cuda_tasks(platform: str) -> dict[str, dict[str, str]]:
    # A target's tasks replace the feature's tasks of the same name on that platform, as Pixi resolves them.
    cuda = PIXI["feature"]["cuda"]
    return cuda["tasks"] | cuda.get("target", {}).get(platform, {}).get("tasks", {})


def _cuda_uv_env(tmp_path: Path, uv: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("UV_")}
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"):
        env.pop(key, None)
    # Activation of every feature the cuda environment composes, as Pixi applies it.
    env.update(PIXI.get("activation", {}).get("env", {}))
    for feature in PIXI["environments"]["cuda"]["features"]:
        env.update(PIXI["feature"][feature].get("activation", {}).get("env", {}))
    env.update(UV_CACHE_DIR=str(tmp_path / "cache"), UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never")
    env["PATH"] = str(Path(uv).parent) + os.pathsep + env["PATH"]
    return env


def _run(args: list[str], cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=30, check=False)  # noqa: S603
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _write_wheel(
    directory: Path, name: str, version: str, requires: str | None = None, modules: tuple[str, ...] = ()
) -> None:
    # A tiny wheel exercises real uv selection and re-sync behavior, without a GPU, network access, or a PyTorch
    # download.
    info = f"{name}-{version}.dist-info"
    requirement = f"Requires-Dist: {requires}\n" if requires else ""
    files = dict.fromkeys(modules, "")
    files[f"{info}/METADATA"] = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{requirement}"
    files[f"{info}/WHEEL"] = "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    files[f"{info}/RECORD"] = "".join(f"{file},,\n" for file in [*files, f"{info}/RECORD"])
    with ZipFile(directory / f"{name}-{version}-py3-none-any.whl", "w") as wheel:
        for file, content in files.items():
            wheel.writestr(file, content)


def _write_probe_project(directory: Path, dependency: str, version: str) -> None:
    (directory / "pyproject.toml").write_text(
        '[project]\nname = "pixi-task-probe"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
        f'dependencies = ["{dependency}"]\n[project.optional-dependencies]\ndev = []\ndocs = []\n'
        f'[tool.uv.sources]\n{dependency} = {{path = "{dependency}-{version}-py3-none-any.whl"}}\n',
        encoding="utf-8",
    )


def test_locked_linux_torch_is_a_cuda_build() -> None:
    # The Linux install task only syncs the lock, so it is only a CUDA install while this holds.
    assert any(
        dependency["name"].startswith("nvidia-cudnn") and "sys_platform == 'linux'" in dependency.get("marker", "")
        for dependency in LOCKED_TORCH["dependencies"]
    )


@pytest.mark.parametrize("task", ["install", "install-docs"])
def test_cuda_index_is_a_url(task: str) -> None:
    args = shlex.split(_cuda_tasks("win-64")[task]["cmd"])
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cuda_index = next(
        index for index in project["tool"]["uv"]["index"] if index["name"] == f"pytorch-{WINDOWS_CUDA_BUILD}"
    )
    # --index takes a URL, not the configured index's name.
    assert args[args.index("--index") + 1] == cuda_index["url"]


@pytest.mark.parametrize(
    ("platform", "expected_torch"),
    [
        ("linux-64", LOCKED_TORCH["version"]),
        ("linux-aarch64", LOCKED_TORCH["version"]),
        ("win-64", f"{LOCKED_TORCH['version']}+{WINDOWS_CUDA_BUILD}"),
    ],
)
@pytest.mark.parametrize("task", ["install", "install-docs"])
@pytest.mark.parametrize("activation", ["CONDA_PREFIX", "VIRTUAL_ENV"])
def test_cuda_install_and_run_share_environment(
    tmp_path: Path, task: str, activation: str, platform: str, expected_torch: str
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required; run this test through Pixi")
    env = _cuda_uv_env(tmp_path, uv)
    project_env = env.get("UV_PROJECT_ENVIRONMENT", ".venv")
    run = partial(_run, cwd=tmp_path, env=env)

    # The index build of the locked version requires a package nothing provides, as a stand-in for the index's
    # dependencies replacing the locked ones. Its newer build must not be selected at all: an index moves on
    # independently of the lock.
    locked = LOCKED_TORCH["version"]
    index_build = f"{locked}+{WINDOWS_CUDA_BUILD}"
    newer_index_build = f"99.0.0+{WINDOWS_CUDA_BUILD}"
    _write_wheel(tmp_path, "torch", locked)
    _write_wheel(tmp_path, "torch", index_build, requires="pixi-task-probe-not-installable")
    _write_wheel(tmp_path, "torch", newer_index_build)
    _write_probe_project(tmp_path, "torch", locked)
    index = tmp_path / "simple" / "torch"
    index.mkdir(parents=True)
    (index / "index.html").write_text(
        "".join(
            f'<a href="../../torch-{version.replace("+", "%2B")}-py3-none-any.whl">torch</a>\n'
            for version in (index_build, newer_index_build)
        ),
        encoding="utf-8",
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
    for command in _cuda_tasks(platform)[task]["cmd"].split(" && "):
        args = shlex.split(command)
        if "--index" in args:
            args[args.index("--index") + 1] = index.parent.as_uri()
        run([uv, *args[1:]])

    # Replace only pytest with a metadata probe; preserve every uv run option
    # from the CUDA test tasks and the inherited focused-test task.
    probe = "import json,sys; from importlib.metadata import version; print(json.dumps([sys.prefix,version('torch')]))"
    tasks = PIXI["tasks"] | _cuda_tasks(platform)
    for name in ("test-cuda", "test-cuda-f32", "test-cuda-f64", "test-cuda-half", "test-module"):
        args = shlex.split(tasks[name]["cmd"])
        env.update(tasks[name].get("env", {}))
        prefix, version = json.loads(run([uv, *args[1 : args.index("pytest")], "python", "-c", probe]))
        assert Path(prefix) == tmp_path / project_env
        assert version == expected_torch
    assert json.loads(run([uv, "pip", "list", "--python", str(pixi_env), "--format", "json"])) == []


@pytest.mark.parametrize("task", ["install", "install-docs"])
def test_cuda_install_restores_files_shared_with_a_stale_wheel(tmp_path: Path, task: str) -> None:
    # A wheel left by an older install can own paths a locked wheel also owns, as nvidia-*-cu12 and nvidia-*-cu13 do.
    # Removing the stale wheel deletes those files while uv still counts the locked wheel as installed, and
    # `import torch` then fails on libcudnn.so.9.
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required; run this test through Pixi")
    env = _cuda_uv_env(tmp_path, uv)
    project_env = env.get("UV_PROJECT_ENVIRONMENT", ".venv")
    run = partial(_run, cwd=tmp_path, env=env)
    _write_wheel(tmp_path, "lockedlibs", "1.0.0", modules=("shared_library.py",))
    _write_wheel(tmp_path, "stalelibs", "1.0.0", modules=("shared_library.py",))
    _write_probe_project(tmp_path, "lockedlibs", "1.0.0")
    run([uv, "venv", "--python", sys.executable, project_env])
    install = [[uv, *shlex.split(command)[1:]] for command in _cuda_tasks("linux-64")[task]["cmd"].split(" && ")]

    for args in install:
        run(args)
    run([uv, "pip", "install", "--python", project_env, "stalelibs-1.0.0-py3-none-any.whl"])
    for args in install:
        run(args)

    run([uv, "run", "--no-sync", "python", "-c", "import shared_library"])


@pytest.mark.parametrize("environment", ["default", "py311", "py312", "py313", "py314"])
def test_non_cuda_environment_selection(environment: str) -> None:
    env = dict(PIXI.get("activation", {}).get("env", {}))
    for feature in PIXI["environments"][environment]["features"]:
        env.update(PIXI["feature"][feature].get("activation", {}).get("env", {}))
    assert "UV_NO_SYNC" not in env
    expected = ".venv" if environment in ("default", "py311") else f".venv-{environment}"
    assert env.get("UV_PROJECT_ENVIRONMENT", ".venv") == expected
