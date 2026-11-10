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

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch.overrides import _get_current_function_mode_stack
from torch.utils._device import DeviceContext

_MISSING = object()


@contextmanager
def default_device(device: str | torch.device) -> Iterator[None]:
    """Run a block under ``torch.set_default_device(device)`` and restore the exact prior state afterwards.

    ``torch.set_default_device`` pushes a ``DeviceContext`` torch-function mode that nothing pops. The obvious
    restore, ``torch.set_default_device(torch.get_default_device())``, puts one ``DeviceContext(cpu)`` in place of
    none when no default was set, and it stays on for the rest of the pytest process (#5508, #5916).
    ``torch.set_default_device(None)`` pops it, but on torch 2.5 to 2.9 it also stores ``None`` where
    ``torch.get_default_device()`` expects a context, so every later call raises ``AttributeError``. This manager
    pops the context it pushed, then puts the thread-local back to what it held before: no attribute at all,
    ``None``, or a context for the same device.

    It refuses to run inside ``with torch.device(...)`` (``RuntimeError``, state untouched): on torch 2.9.1 or older
    ``DeviceContext.__exit__`` asserts on the mode it finds on top of the stack, which corrupts the state. torch 2.14
    handles that nesting, but the helper rejects it on every version so a test cannot pass on one and break on another.
    """
    device = torch.device(device)  # validate first: torch pops the prior context before it checks the new device
    store = torch._GLOBAL_DEVICE_CONTEXT
    prior = getattr(store, "device_context", _MISSING)
    foreign = [
        mode for mode in _get_current_function_mode_stack() if isinstance(mode, DeviceContext) and mode is not prior
    ]
    if foreign:
        raise RuntimeError("default_device cannot be used inside `with torch.device(...)`")
    prior_device = None if prior is _MISSING or prior is None else prior.device
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(None)  # pops the context pushed above, leaves ``None`` behind
        if prior_device is not None:
            # The first call popped ``prior``; push a fresh context for its device.
            torch.set_default_device(prior_device)
        elif prior is _MISSING:
            del store.device_context
