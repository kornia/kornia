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
import copy
import doctest
import importlib
import inspect
import math
import os
import pickle
import subprocess
import sys
import types

import pytest

from kornia.config import LazyLoaderConfig, kornia_config
from kornia.core import external
from kornia.core.external import LazyLoader

MISSING = "kornia_test_no_such_module_xyz"


class TestLazyLoader:
    def test_lazy_loader_initialization(self):
        # Test that the LazyLoader initializes with the correct module name and None module
        loader = LazyLoader("math")
        assert loader.module_name == "math"
        assert loader.module is None

    def test_lazy_loader_loading_module(self):
        # Test that the LazyLoader correctly loads the module upon attribute access
        loader = LazyLoader("math")
        assert loader.module is None  # Should be None before any attribute access

        # Access an attribute to trigger module loading
        assert loader.sqrt(4) == 2.0
        assert loader.module is not None  # Should be loaded now

    def test_lazy_loader_invalid_module(self):
        # Test that LazyLoader raises an ImportError for an invalid module (the default mode raises; nothing is asked)
        loader = LazyLoader("non_existent_module")
        with pytest.raises(ImportError) as excinfo:
            _ = loader.non_existent_attribute  # Accessing any attribute should raise the error

        assert "Optional dependency 'non_existent_module' is not installed" in str(excinfo.value)

    def test_lazy_loader_getattr(self):
        # Test that __getattr__ works correctly for a valid module
        loader = LazyLoader("math")
        assert loader.sqrt(16) == 4.0
        assert loader.pi == 3.141592653589793

    def test_lazy_loader_dir(self):
        # Test that dir() returns the correct list of attributes for the module
        loader = LazyLoader("math")
        attributes = dir(loader)
        assert "sqrt" in attributes
        assert "pi" in attributes
        assert loader.module is not None

    def test_lazy_loader_multiple_attributes(self):
        # Test accessing multiple attributes to ensure the module is loaded only once
        loader = LazyLoader("math")
        assert loader.sqrt(25) == 5.0
        assert loader.pi == 3.141592653589793
        assert loader.pow(2, 3) == 8.0
        assert loader.module is not None

    def test_lazy_loader_non_existing_attribute(self):
        # Test that accessing a non-existing attribute raises an AttributeError after loading
        loader = LazyLoader("math")
        with pytest.raises(AttributeError):
            _ = loader.non_existent_attribute


@pytest.fixture
def untouchable(monkeypatch):
    """Put the loaders in AUTO mode and fail on any prompt or process start: a probe must do neither."""
    previous = kornia_config.lazyloader.installation_mode
    kornia_config.lazyloader.installation_mode = "auto"

    def forbidden(*args, **kwargs):
        raise AssertionError(f"a lazy-loader probe tried to prompt or start a process: {args!r}")

    for name in ("run", "call", "check_call", "check_output", "Popen"):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(builtins, "input", forbidden)
    yield
    kornia_config.lazyloader.installation_mode = previous


class TestLazyLoaderCopyAndPickle:
    """A loader survives copy, deepcopy and pickling, before and after it has loaded its module."""

    @pytest.mark.parametrize("loaded", [False, True])
    @pytest.mark.parametrize(
        "duplicate",
        # S301: a round trip of an object this test just built.
        [copy.copy, copy.deepcopy, lambda loader: pickle.loads(pickle.dumps(loader))],  # noqa: S301
        ids=["copy", "deepcopy", "pickle"],
    )
    def test_duplicate_is_a_working_loader(self, duplicate, loaded):
        loader = LazyLoader("math", extra="image")
        if loaded:
            assert loader.pi == math.pi
        other = duplicate(loader)
        assert isinstance(other, LazyLoader)
        assert (other.module_name, other.dev_dependency, other.extra) == ("math", False, "image")
        assert other.pi == math.pi
        assert other.sqrt(16) == 4.0

    def test_copy_of_a_missing_module_loader(self, untouchable):
        other = copy.copy(LazyLoader(MISSING, extra="sd"))
        assert other.module_name == MISSING
        assert other.module is None


class TestLazyLoaderProtocolProbes:
    """Protocol lookups (dunder names) never import, install or prompt for a module that is not loaded."""

    def test_probes_on_a_missing_module(self, untouchable):
        loader = LazyLoader(MISSING, extra="sd")
        assert not hasattr(loader, "__wrapped__")
        assert inspect.unwrap(loader) is loader
        assert not inspect.isroutine(loader)
        assert loader.module is None

    def test_doctest_finder_skips_a_missing_module_loader(self, untouchable):
        # What ``pytest --doctest-modules`` does to every module: walk its namespace for routines and classes.
        module = types.ModuleType("kornia_test_doctest_module")

        def documented():
            """Return one.

            >>> documented()
            1
            """
            return 1

        # Rebind the function to the module's globals, as if it were defined there.
        module.documented = types.FunctionType(documented.__code__, module.__dict__, "documented")
        module.optional = LazyLoader(MISSING, extra="sd")
        tests = doctest.DocTestFinder().find(module)
        assert [test.name for test in tests] == ["kornia_test_doctest_module.documented"]
        assert module.optional.module is None

    @pytest.mark.parametrize(
        ("module_name", "attribute"),
        [("numpy", "__version__"), ("json", "__file__"), ("json", "__path__"), ("json", "__all__")],
    )
    def test_module_metadata_loads_the_module(self, module_name, attribute):
        module = importlib.import_module(module_name)
        loader = LazyLoader(module_name)
        assert getattr(loader, attribute) == getattr(module, attribute)
        assert loader.module is module

    def test_module_metadata_of_a_missing_module_raises_import_error(self, monkeypatch):
        monkeypatch.delenv("KORNIA_INSTALLATION_MODE", raising=False)
        monkeypatch.setattr(kornia_config, "lazyloader", LazyLoaderConfig())
        with pytest.raises(ImportError):
            _ = LazyLoader(MISSING, extra="sd").__version__

    def test_probe_forwards_to_a_loaded_module(self):
        loader = LazyLoader("math")
        assert loader.pi == math.pi
        assert loader.__name__ == "math"


class TestLazyLoaderUnderDoctestModules:
    """``--doctest-modules`` in ``sys.argv`` does not stop an installed module from loading."""

    @pytest.mark.parametrize("dev_dependency", [False, True])
    def test_installed_module_loads(self, monkeypatch, dev_dependency):
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--doctest-modules"])
        assert LazyLoader("math", dev_dependency=dev_dependency).pi == math.pi

    def test_installed_onnxruntime_loads(self, monkeypatch):
        onnxruntime = pytest.importorskip("onnxruntime")
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--doctest-modules"])
        declared = external.onnxruntime
        loader = LazyLoader(declared.module_name, dev_dependency=declared.dev_dependency, extra=declared.extra)
        assert loader.InferenceSession is onnxruntime.InferenceSession
