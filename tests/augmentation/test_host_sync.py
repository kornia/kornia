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

"""CUDA host-synchronization guard for every augmentation in ``testing.augmentation.sync_registry``."""

from __future__ import annotations

import inspect

import pytest
import torch

import kornia.augmentation as K
from kornia.augmentation import auto as KAuto
from kornia.augmentation import container as KContainer
from kornia.augmentation import presets as KPresets
from kornia.augmentation.base import _BasicAugmentationBase
from kornia.augmentation.container import ImageSequential
from kornia.augmentation.presets import ada as KAda

from testing.augmentation.sync_registry import AUG_CASES, EXCLUDED, INHERENT_SITES, make_call, sync_sites
from testing.base import BaseTester

_CALLS = [pytest.param(case, which, id=f"{case.name}-{which}") for case in AUG_CASES for which in case.cases]
_HALF = (torch.float16, torch.bfloat16)


def test_registry_covers_public_augmentations() -> None:
    public = set()
    for module in (K, KAuto, KContainer, KPresets, KAda):
        for name in dir(module):
            obj = getattr(module, name)
            if name.startswith("_"):
                continue
            if inspect.isclass(obj) and issubclass(obj, (_BasicAugmentationBase, ImageSequential)):
                public.add(obj.__name__)
    registered = {case.name for case in AUG_CASES}
    missing = sorted(public - registered - EXCLUDED.keys())
    assert not missing, f"Register these augmentations in testing/augmentation/sync_registry.py: {missing}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host synchronizations need a CUDA device")
class TestAugmentationHostSync(BaseTester):
    @pytest.mark.parametrize("case, which", _CALLS)
    def test_no_avoidable_host_sync(self, case, which, device, dtype) -> None:
        if device.type != "cuda":
            pytest.skip("CUDA host synchronizations need a CUDA device")
        fn = make_call(case, which, device, dtype)
        try:
            fn()
        except (RuntimeError, NotImplementedError, TypeError) as error:
            if dtype not in _HALF:
                raise
            pytest.skip(f"{case.name} does not run in {dtype}: {error}")
        torch.cuda.synchronize()
        sites = sync_sites(fn)
        assert not set(sites) - INHERENT_SITES.keys(), dict(sites)
