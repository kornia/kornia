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

import builtins
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

import pytest

from kornia.config import InstallationMode, LazyLoaderConfig, kornia_config
from kornia.core import external
from kornia.core.external import LazyLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_VAR = "KORNIA_INSTALLATION_MODE"
# A module name that no environment provides, so "missing" never depends on what is installed.
MISSING = "kornia_test_no_such_module_xyz"


class InstallRefused(Exception):
    """Raised by the process-start stubs: no test may run pip or any other command."""


@pytest.fixture
def restore_mode():
    """Put the process-wide installation mode back after a test changes it."""
    previous = kornia_config.lazyloader.installation_mode
    yield
    kornia_config.lazyloader.installation_mode = previous


@pytest.fixture
def commands(monkeypatch):
    """Record every attempt to start a process and refuse it, so no test can install anything."""
    recorded = []

    def refuse(name):
        def start(*args, **kwargs):
            recorded.append(args[0] if args else kwargs.get("args"))
            raise InstallRefused(name)

        return start

    for name in ("run", "call", "check_call", "check_output", "Popen"):
        monkeypatch.setattr(subprocess, name, refuse(f"subprocess.{name}"))
    monkeypatch.setattr(os, "system", refuse("os.system"))
    return recorded


@pytest.fixture
def no_prompt(monkeypatch):
    """Fail the test if the loader asks anything."""

    def forbidden_input(prompt=""):
        raise AssertionError(f"the loader prompted: {prompt!r}")

    monkeypatch.setattr(builtins, "input", forbidden_input)


@pytest.fixture
def blocked(monkeypatch):
    """Make the loaders see the module names added to the returned set as not installed."""
    names = set()
    real_import = external.importlib.import_module

    def import_module(name, package=None):
        if name in names:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, package)

    fake = types.SimpleNamespace(import_module=import_module, invalidate_caches=lambda: None)
    monkeypatch.setattr(external, "importlib", fake)
    return names


def pip_install(extra):
    return [sys.executable, "-m", "pip", "install", f"kornia[{extra}]"]


def not_installed(module_name, extra=None):
    hint = (
        f'Install it with: pip install "kornia[{extra}]".' if extra else "Please install it to use this functionality."
    )
    return f"Optional dependency '{module_name}' is not installed. {hint}"


class FakeTerminal:
    """A stdin that reports an interactive terminal; the answers come from a patched ``input``."""

    def isatty(self):
        return True


@pytest.fixture
def answers(monkeypatch):
    """Answer the loader's questions on a pretend terminal; returns (answers to give, prompts seen)."""
    to_give, prompts = [], []

    def scripted_input(prompt=""):
        prompts.append(prompt)
        if not to_give:
            raise EOFError
        return to_give.pop(0)

    monkeypatch.setattr(sys, "stdin", FakeTerminal())
    monkeypatch.setattr(builtins, "input", scripted_input)
    return to_give, prompts


class TestLazyLoaderExtra:
    """Check that a missing optional dependency points at the extra that installs it."""

    def test_missing_dependency_names_the_extra(self, restore_mode):
        kornia_config.lazyloader.installation_mode = InstallationMode.RAISE
        loader = LazyLoader("definitely_not_a_module_xyz", extra="sd")
        with pytest.raises(ImportError) as excinfo:
            loader.__getattr__("x")

        message = str(excinfo.value)
        assert "Optional dependency 'definitely_not_a_module_xyz' is not installed" in message
        assert 'pip install "kornia[sd]"' in message

    def test_missing_dependency_without_extra_keeps_generic_message(self, restore_mode):
        kornia_config.lazyloader.installation_mode = InstallationMode.RAISE
        loader = LazyLoader("definitely_not_a_module_xyz")
        with pytest.raises(ImportError) as excinfo:
            loader.__getattr__("x")

        message = str(excinfo.value)
        assert "Please install it to use this functionality." in message
        assert "kornia[" not in message

    @pytest.mark.parametrize(
        ("name", "extra"),
        [
            ("PILImage", "image"),
            ("onnx", "onnx"),
            ("onnxruntime", "onnx"),
            ("diffusers", "sd"),
        ],
    )
    def test_declared_loaders_carry_their_extra(self, name, extra):
        assert getattr(external, name).extra == extra


class TestInstallationModeConfig:
    """The default mode, the KORNIA_INSTALLATION_MODE environment variable and the setter."""

    def test_default_mode_is_raise(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert LazyLoaderConfig().installation_mode is InstallationMode.RAISE

    def test_default_mode_raises_import_error_without_prompting(self, monkeypatch, commands, no_prompt):
        monkeypatch.delenv(ENV_VAR, raising=False)
        monkeypatch.setattr(kornia_config, "lazyloader", LazyLoaderConfig())
        monkeypatch.setattr(sys, "stdin", FakeTerminal())
        with pytest.raises(ImportError) as excinfo:
            _ = LazyLoader(MISSING, extra="image").attr
        assert str(excinfo.value) == not_installed(MISSING, "image")
        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)
        assert commands == []

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("raise", InstallationMode.RAISE),
            ("ask", InstallationMode.ASK),
            ("auto", InstallationMode.AUTO),
            ("AUTO", InstallationMode.AUTO),
            ("Ask", InstallationMode.ASK),
        ],
    )
    def test_env_var_sets_the_mode(self, monkeypatch, value, expected):
        monkeypatch.setenv(ENV_VAR, value)
        assert LazyLoaderConfig().installation_mode is expected

    @pytest.mark.parametrize("value", ["", "  "])
    def test_empty_env_var_keeps_the_default(self, monkeypatch, value):
        monkeypatch.setenv(ENV_VAR, value)
        assert LazyLoaderConfig().installation_mode is InstallationMode.RAISE

    @staticmethod
    def _assert_names_the_bad_value(message, value):
        assert ENV_VAR in message
        assert repr(value) in message
        for choice in ("'raise'", "'ask'", "'auto'"):
            assert choice in message

    def test_invalid_env_var_is_reported_when_the_mode_is_needed(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "maybe")
        config = LazyLoaderConfig()  # creating the config, as `import kornia` does, does not raise
        with pytest.raises(ValueError) as excinfo:
            _ = config.installation_mode
        self._assert_names_the_bad_value(str(excinfo.value), "maybe")
        # A mode set in code replaces the invalid value.
        config.installation_mode = "raise"
        assert config.installation_mode is InstallationMode.RAISE

    @pytest.mark.parametrize("extra", [None, "image"])
    def test_invalid_env_var_is_reported_by_a_loader_for_a_missing_module(
        self, monkeypatch, commands, no_prompt, extra
    ):
        monkeypatch.setenv(ENV_VAR, "maybe")
        monkeypatch.setattr(kornia_config, "lazyloader", LazyLoaderConfig())
        assert LazyLoader("math").pi == pytest.approx(3.141592653589793)  # an installed module does not need the mode
        with pytest.raises(ValueError) as excinfo:
            _ = LazyLoader(MISSING, extra=extra).attr
        self._assert_names_the_bad_value(str(excinfo.value), "maybe")
        assert commands == []

    def test_invalid_env_var_does_not_break_import_kornia(self):
        env = {**{k: v for k, v in os.environ.items() if k != ENV_VAR}, ENV_VAR: "bogus"}
        code = textwrap.dedent(
            f"""
            import os
            import subprocess

            import kornia
            from kornia.core.external import LazyLoader

            def refuse(*args, **kwargs):
                raise RuntimeError("process start refused")

            for name in ("run", "call", "check_call", "check_output", "Popen"):
                setattr(subprocess, name, refuse)
            os.system = refuse

            print("IMPORTED", kornia.__name__, flush=True)
            print("INSTALLED", LazyLoader("math").pi, flush=True)
            try:
                LazyLoader({MISSING!r}, extra="image").attr
            except BaseException as e:
                print("MISSING", type(e).__name__, str(e), flush=True)
            """
        )
        # S603: this interpreter and a literal program; cwd is the repo root so the child imports this tree.
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert "IMPORTED kornia" in lines, result.stdout
        assert "INSTALLED 3.141592653589793" in lines, result.stdout
        missing = [line for line in lines if line.startswith("MISSING ")]
        assert len(missing) == 1, result.stdout
        assert missing[0].startswith("MISSING ValueError "), missing[0]
        self._assert_names_the_bad_value(missing[0], "bogus")

    @pytest.mark.parametrize(("value", "expected"), [(None, "RAISE"), ("auto", "AUTO")])
    def test_global_config_reads_the_env_var_at_import(self, value, expected):
        env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
        if value is not None:
            env[ENV_VAR] = value
        code = (
            "from kornia.config import kornia_config\nprint('MODE', kornia_config.lazyloader.installation_mode.value)"
        )
        # S603: this interpreter and a literal program; cwd is the repo root so the child imports this tree.
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert f"MODE {expected}" in result.stdout.splitlines()

    @pytest.mark.parametrize("value", ["raise", "Ask", "AUTO", InstallationMode.ASK])
    def test_setter_accepts_any_case(self, restore_mode, value):
        kornia_config.lazyloader.installation_mode = value
        assert kornia_config.lazyloader.installation_mode is InstallationMode(str(value).upper())

    def test_setter_rejects_unknown_values(self, restore_mode):
        with pytest.raises(ValueError) as excinfo:
            kornia_config.lazyloader.installation_mode = "maybe"
        for choice in ("'raise'", "'ask'", "'auto'"):
            assert choice in str(excinfo.value)
        with pytest.raises(TypeError):
            kornia_config.lazyloader.installation_mode = None


class TestInstallationModeProtocol:
    """InstallationMode hashes, and ``==`` / ``!=`` agree, for members and strings."""

    def test_members_are_hashable(self):
        table = {mode: mode.value for mode in InstallationMode}
        assert table[InstallationMode.ASK] == "ASK"
        assert {InstallationMode.RAISE, InstallationMode.RAISE} == {InstallationMode.RAISE}

    @pytest.mark.parametrize("mode", list(InstallationMode))
    @pytest.mark.parametrize("other", ["ASK", "ask", "AUTO", "auto", "RAISE", "raise", "other", *InstallationMode])
    def test_eq_ne_and_hash_agree(self, mode, other):
        assert (mode == other) is not (mode != other)
        assert (other == mode) is not (other != mode)
        if mode == other:
            assert hash(mode) == hash(other)


class TestAskMode:
    """``ASK`` asks only on an interactive terminal and otherwise behaves as ``RAISE``."""

    def test_non_terminal_stdin_raises_import_error(self, monkeypatch, restore_mode, commands, no_prompt):
        kornia_config.lazyloader.installation_mode = "ask"
        # A readable stdin that would answer "yes" if the loader read it; it is not a terminal.
        monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
        with pytest.raises(ImportError) as excinfo:
            _ = LazyLoader(MISSING, extra="image").attr
        assert str(excinfo.value) == not_installed(MISSING, "image")
        assert commands == []

    @staticmethod
    def _child_code():
        return textwrap.dedent(
            f"""
            import os
            import subprocess

            from kornia.config import kornia_config
            from kornia.core.external import LazyLoader

            recorded = []

            def refuse(name):
                def start(*args, **kwargs):
                    recorded.append(name)
                    raise RuntimeError(name + " refused")
                return start

            for name in ("run", "call", "check_call", "check_output", "Popen"):
                setattr(subprocess, name, refuse("subprocess." + name))
            os.system = refuse("os.system")

            kornia_config.lazyloader.installation_mode = "ask"
            try:
                _ = LazyLoader({MISSING!r}, extra="image").attr
                print("RESULT returned", flush=True)
            except BaseException as e:
                print("RESULT", type(e).__name__, isinstance(e, ImportError), flush=True)
            print("RECORDED", recorded, flush=True)
            """
        )

    def test_devnull_stdin_raises_import_error_in_a_child_process(self):
        env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
        # S603: this interpreter and a literal program; cwd is the repo root so the child imports this tree.
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", self._child_code()],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert "RESULT ImportError True" in lines, result.stdout
        assert "RECORDED []" in lines, result.stdout

    def test_open_silent_stdin_does_not_block(self):
        env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
        with tempfile.TemporaryFile(mode="w+") as out:
            # S603: this interpreter and a literal program. stdin is a pipe that stays open and is never written,
            # as some job runners and IDE consoles leave it; a loader that reads it would block forever.
            child = subprocess.Popen(  # noqa: S603
                [sys.executable, "-c", self._child_code()],
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                pytest.fail("the loader blocked reading an open stdin that is not a terminal")
            finally:
                child.stdin.close()
            out.seek(0)
            output = out.read()
        assert child.returncode == 0, output
        lines = output.splitlines()
        assert "RESULT ImportError True" in lines, output
        assert "RECORDED []" in lines, output

    def test_terminal_answer_no_raises_import_error(self, restore_mode, commands, answers):
        to_give, prompts = answers
        to_give.append("n")
        kornia_config.lazyloader.installation_mode = "ask"
        with pytest.raises(ImportError) as excinfo:
            _ = LazyLoader(MISSING, extra="image").attr
        assert str(excinfo.value) == not_installed(MISSING, "image")
        assert len(prompts) == 1
        assert 'pip install "kornia[image]"' in prompts[0]
        assert "'raise'" in prompts[0]
        assert ENV_VAR in prompts[0]
        assert commands == []

    @pytest.mark.parametrize("answer", ["y", "YES"])
    def test_terminal_answer_yes_installs_the_extra(self, restore_mode, commands, answers, answer):
        to_give, prompts = answers
        to_give.append(answer)
        kornia_config.lazyloader.installation_mode = "ask"
        with pytest.raises(InstallRefused):
            _ = LazyLoader(MISSING, extra="image").attr
        assert commands == [pip_install("image")]
        assert len(prompts) == 1
        assert kornia_config.lazyloader.installation_mode is InstallationMode.ASK

    def test_terminal_answer_all_switches_the_process_to_auto(self, restore_mode, commands, answers):
        to_give, prompts = answers
        to_give.append("a")
        kornia_config.lazyloader.installation_mode = "ask"
        with pytest.raises(InstallRefused):
            _ = LazyLoader(MISSING, extra="image").attr
        assert kornia_config.lazyloader.installation_mode is InstallationMode.AUTO
        # Another loader, for another extra, installs without asking again.
        with pytest.raises(InstallRefused):
            _ = LazyLoader(MISSING + "_2", extra="sd").attr
        assert len(prompts) == 1
        assert commands == [pip_install("image"), pip_install("sd")]

    def test_terminal_invalid_answer_asks_again(self, restore_mode, commands, answers):
        to_give, prompts = answers
        to_give.extend(["maybe", "n"])
        kornia_config.lazyloader.installation_mode = "ask"
        with pytest.raises(ImportError):
            _ = LazyLoader(MISSING, extra="image").attr
        assert len(prompts) == 2
        assert commands == []

    def test_terminal_end_of_input_raises_import_error(self, restore_mode, commands, answers):
        _, prompts = answers  # no answer: input() raises EOFError
        kornia_config.lazyloader.installation_mode = "ask"
        with pytest.raises(ImportError) as excinfo:
            _ = LazyLoader(MISSING, extra="image").attr
        assert str(excinfo.value) == not_installed(MISSING, "image")
        assert len(prompts) == 1
        assert commands == []

    def test_terminal_does_not_ask_for_a_loader_without_extra(self, restore_mode, commands, answers):
        to_give, prompts = answers
        to_give.append("y")
        kornia_config.lazyloader.installation_mode = "ask"
        with pytest.raises(ImportError) as excinfo:
            _ = LazyLoader(MISSING).attr
        assert str(excinfo.value) == not_installed(MISSING)
        assert prompts == []
        assert commands == []


class TestAutoMode:
    """``AUTO`` installs the declared kornia extra, and nothing for a loader without one."""

    @pytest.mark.parametrize(
        ("name", "extra"),
        [("PILImage", "image"), ("onnx", "onnx"), ("onnxruntime", "onnx"), ("diffusers", "sd")],
    )
    def test_installs_the_declared_extra(self, restore_mode, commands, no_prompt, blocked, name, extra):
        declared = getattr(external, name)
        loader = LazyLoader(declared.module_name, dev_dependency=declared.dev_dependency, extra=declared.extra)
        blocked.add(declared.module_name)
        kornia_config.lazyloader.installation_mode = "auto"
        with pytest.raises(InstallRefused):
            _ = loader.attr
        assert commands == [pip_install(extra)]

    @pytest.mark.parametrize("module_name", [MISSING, "numpy"])
    def test_loader_without_extra_never_installs(self, restore_mode, commands, no_prompt, blocked, module_name):
        blocked.add(module_name)
        kornia_config.lazyloader.installation_mode = "auto"
        with pytest.raises(ImportError) as excinfo:
            _ = LazyLoader(module_name).attr
        assert str(excinfo.value) == not_installed(module_name)
        assert commands == []

    def test_pip_failure_raises_import_error_once(self, monkeypatch, restore_mode, commands, no_prompt):
        def failing_pip(args, *rest, check=False, **kwargs):
            commands.append(args)
            if check:
                raise subprocess.CalledProcessError(1, args)
            return subprocess.CompletedProcess(args, 1)

        monkeypatch.setattr(subprocess, "run", failing_pip)
        kornia_config.lazyloader.installation_mode = "auto"
        loader = LazyLoader(MISSING, extra="sd")
        with pytest.raises(ImportError) as excinfo:
            _ = loader.attr
        assert 'pip install "kornia[sd]"' in str(excinfo.value)
        assert "failed" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)
        # The failure is remembered: the next access raises again without running pip again.
        with pytest.raises(ImportError, match="failed"):
            _ = loader.attr
        assert commands == [pip_install("sd")]

    def test_successful_install_imports_the_module(self, monkeypatch, restore_mode, commands, no_prompt, blocked):
        blocked.add("math")

        def working_pip(args, *rest, check=False, **kwargs):
            commands.append(args)
            blocked.discard("math")  # what the installation would make importable
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(subprocess, "run", working_pip)
        kornia_config.lazyloader.installation_mode = "auto"
        loader = LazyLoader("math", extra="image")
        assert loader.pi == pytest.approx(3.141592653589793)
        assert commands == [pip_install("image")]
