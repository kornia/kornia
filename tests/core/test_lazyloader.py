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

from io import StringIO

import pytest

from kornia.core.external import LazyLoader


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

    def test_lazy_loader_invalid_module(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO("n"))
        # Test that LazyLoader raises an ImportError for an invalid module
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
