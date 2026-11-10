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

import pytest
import torch
from torch.overrides import _get_current_function_mode_stack
from torch.utils._device import DeviceContext

from testing.default_device import default_device


def _device_contexts():
    return [mode for mode in _get_current_function_mode_stack() if isinstance(mode, DeviceContext)]


@pytest.fixture
def clean_device_state():
    """Run a test from a pristine default-device state (no attribute on the thread-local) and put it back."""
    store = torch._GLOBAL_DEVICE_CONTEXT
    had_attribute = hasattr(store, "device_context")
    saved = getattr(store, "device_context", None)
    saved_device = None if saved is None else saved.device
    torch.set_default_device(None)
    if hasattr(store, "device_context"):
        del store.device_context
    try:
        yield
    finally:
        torch.set_default_device(None)
        if saved_device is not None:
            torch.set_default_device(saved_device)
        elif not had_attribute and hasattr(store, "device_context"):
            del store.device_context  # originally absent; an originally-``None`` attribute stays ``None``


@pytest.mark.device_agnostic
class TestDefaultDevice:
    def test_default_is_applied_inside_and_state_is_restored(self, clean_device_state):
        stack_before = list(_get_current_function_mode_stack())
        with default_device("meta"):
            assert torch.get_default_device().type == "meta"
            assert torch.empty(1).device.type == "meta"
            assert len(_device_contexts()) == 1
        assert list(_get_current_function_mode_stack()) == stack_before
        assert not _device_contexts()
        assert torch.get_default_device() == torch.device("cpu")
        assert torch.empty(1).device.type == "cpu"
        # Exact pre-block state: no thread-local attribute (a ``None`` there breaks get_default_device on torch <= 2.9).
        assert not hasattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context")

    def test_restores_after_an_exception(self, clean_device_state):
        with pytest.raises(RuntimeError, match="boom"), default_device("meta"):
            raise RuntimeError("boom")
        assert not _device_contexts()
        assert torch.get_default_device() == torch.device("cpu")

    def test_previously_set_default_survives(self, clean_device_state):
        torch.set_default_device("meta")
        try:
            with default_device("cpu"):
                assert torch.get_default_device().type == "cpu"
            assert torch.get_default_device().type == "meta"
            assert len(_device_contexts()) == 1
            assert torch.empty(1).device.type == "meta"
        finally:
            torch.set_default_device(None)
        assert not _device_contexts()

    def test_nested(self, clean_device_state):
        with default_device("meta"):
            with default_device("cpu"):
                assert torch.get_default_device().type == "cpu"
            assert torch.get_default_device().type == "meta"
            assert len(_device_contexts()) == 1
        assert not _device_contexts()
        assert torch.get_default_device() == torch.device("cpu")

    def test_invalid_device_with_a_prior_default_leaves_state_intact(self, clean_device_state):
        torch.set_default_device("meta")
        try:
            with pytest.raises((RuntimeError, ValueError)), default_device("not-a-device"):
                pass  # pragma: no cover
            assert torch.get_default_device().type == "meta"
            assert len(_device_contexts()) == 1
            assert torch.empty(1).device.type == "meta"
        finally:
            torch.set_default_device(None)
        assert not _device_contexts()

    def test_refuses_to_nest_inside_torch_device(self, clean_device_state):
        with torch.device("cpu"):
            outer = list(_get_current_function_mode_stack())
            with pytest.raises(RuntimeError, match="cannot be used inside"), default_device("meta"):
                pass  # pragma: no cover
            assert list(_get_current_function_mode_stack()) == outer
        assert not _device_contexts()
        assert not hasattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context")
