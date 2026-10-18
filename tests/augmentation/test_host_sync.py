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
import json

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

# Augmentations whose stored ``_params`` are replayed, as given and moved to CUDA (Review Focus 1).
_REPLAY = {
    "RandomAffine": lambda: K.RandomAffine(15.0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10.0, p=1.0),
    "RandomBrightness": lambda: K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
    "RandomCrop": lambda: K.RandomCrop((16, 16), p=1.0),
    "RandomPerspective": lambda: K.RandomPerspective(0.5, p=1.0),
}
# One augmentation per gate (Review Focus 2): ``_blend_by_prob`` for an intensity and a geometric augmentation, the
# matrix gates of 2D and 3D geometry, and the batch-level gate of the mix augmentations.
_GATED = {
    "RandomAffine": (lambda p, same: K.RandomAffine(30.0, p=p, same_on_batch=same), (8, 3, 16, 16)),
    "RandomBrightness": (lambda p, same: K.RandomBrightness((0.5, 1.5), p=p, same_on_batch=same), (8, 3, 16, 16)),
    "RandomHorizontalFlip3D": (lambda p, same: K.RandomHorizontalFlip3D(p=p, same_on_batch=same), (8, 3, 4, 8, 8)),
    "RandomMixUpV2": (lambda p, same: K.RandomMixUpV2(p=p, same_on_batch=same), (8, 3, 16, 16)),
}
# Mix augmentations run on an empty batch (Review Focus 4).
_EMPTY_BATCH_MIX = {
    "RandomCutMixV2": lambda: K.RandomCutMixV2(p=1.0, use_correct_lambda=True),
    "RandomMixUpV2": lambda: K.RandomMixUpV2(p=1.0),
}
# Host reads that only happen because the caller moved the parameters to CUDA: the crop window and the padding set
# the sizes of Python slices.
_REPLAY_DEVICE_READS = {
    "RandomCrop": {"geometry/transform/crop2d.py:crop_by_indices", "augmentation/_2d/geometric/crop.py:apply_transform"}
}


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

    @pytest.mark.parametrize("placement", ["cpu", "cuda"])
    @pytest.mark.parametrize("name", sorted(_REPLAY))
    def test_replay_params_cuda_and_cpu(self, name, placement, device, dtype) -> None:
        if device.type != "cuda":
            pytest.skip("CUDA host synchronizations need a CUDA device")
        op = _REPLAY[name]()
        x = torch.rand(4, 3, 32, 32, device=device, dtype=dtype)
        first = op(x)
        params = {
            k: v.to(device) if placement == "cuda" and isinstance(v, torch.Tensor) else v for k, v in op._params.items()
        }
        torch.cuda.synchronize()
        replayed = []
        sites = sync_sites(lambda: replayed.append(op(x, params=params)))
        assert torch.equal(replayed[0], first)
        # Cropping reads the crop window on the host to slice; that is inherent when the caller moved it to CUDA.
        allowed = INHERENT_SITES.keys() | (_REPLAY_DEVICE_READS.get(name, set()) if placement == "cuda" else set())
        assert not set(sites) - allowed, dict(sites)

    @pytest.mark.parametrize("name", ["RandomCutMixV2", "RandomMixUpV2"])
    def test_empty_batch_mix(self, name, device, dtype) -> None:
        if device.type != "cuda":
            pytest.skip("CUDA host synchronizations need a CUDA device")
        op = _EMPTY_BATCH_MIX[name]()
        x = torch.rand(0, 3, 32, 32, device=device, dtype=dtype)
        op(x)
        torch.cuda.synchronize()
        outputs = []
        sites = sync_sites(lambda: outputs.append(op(x)))
        assert outputs[0].shape == x.shape
        assert outputs[0].device == x.device
        assert not set(sites) - INHERENT_SITES.keys(), dict(sites)

    @pytest.mark.parametrize("same_on_batch", [False, True])
    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("name", sorted(_GATED))
    def test_gate_selection_unchanged(self, name, p, same_on_batch, device, dtype) -> None:
        if device.type != "cuda":
            pytest.skip("CUDA host synchronizations need a CUDA device")
        build, shape = _GATED[name]
        op = build(p, same_on_batch)
        x = torch.rand(*shape, device=device, dtype=dtype)
        op(x)
        torch.cuda.synchronize()
        outputs = []
        sites = sync_sites(lambda: outputs.append(op(x)))
        assert not set(sites) - INHERENT_SITES.keys(), dict(sites)
        params = op._params
        # The gated rows are the always-applied result of the same parameters; the others are the input itself.
        reference = build(1.0, same_on_batch)(x, params=params)
        applied = torch.atleast_1d(params["batch_prob"] > 0.5).to(device)
        if applied.shape[0] != x.shape[0]:  # a batch-level gate, as the mix augmentations draw it
            applied = applied.expand(x.shape[0])
        expected = torch.where(applied.view(-1, *([1] * (x.dim() - 1))), reference, x)
        assert torch.equal(outputs[0], expected)


class TestMaskWritesWithoutHostReads(BaseTester):
    """RandomRain and RandomSaltAndPepperNoise write their pixels without a ``nonzero`` (seeds 0 to 4)."""

    @pytest.mark.parametrize("seed", range(5))
    def test_rain_and_salt_pepper_values_unchanged(self, seed, device, dtype) -> None:
        x = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(seed)).to(device, dtype)

        # Salt and pepper: the same writes as the boolean indexing they replace, salt first.
        torch.manual_seed(seed)
        noise = K.RandomSaltAndPepperNoise(p=1.0)
        out = noise(x)
        expected = x.clone()
        expected[noise._params["mask_salt"].to(device)] = 1.0
        expected[noise._params["mask_pepper"].to(device)] = 0.0
        assert torch.equal(out, expected)

        # Rain: every drop pixel takes 200 / 255 in all channels and every other pixel keeps the input, and the
        # replayed parameters give the same image on the CPU and on the device.
        torch.manual_seed(seed)
        rain = K.RandomRain(p=1.0)
        out = rain(x)
        changed = (out != x).any(dim=1, keepdim=True).expand_as(out)
        assert changed.any()
        drop_value = torch.full((), 200 / 255, dtype=dtype)
        assert torch.equal(out[changed].cpu(), drop_value.expand(int(changed.sum())))
        assert torch.equal(out[~changed], x[~changed])
        params = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in rain._params.items()}
        assert torch.equal(rain(x.cpu(), params=params), out.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host-to-device copies need a CUDA device")
def test_erasing_copies_box_parameters_not_masks(tmp_path) -> None:
    # A large pageable copy stalls the host while the driver stages it, without any cudaStreamSynchronize: the
    # erasing mask is built on the device from the box parameters, so only the parameters cross.
    device = torch.device("cuda")
    op = K.RandomErasing(p=1.0)
    x = torch.rand(8, 3, 64, 64, device=device)
    op(x)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        op(x)
        torch.cuda.synchronize()
    trace = tmp_path / "trace.json"
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    copied = sum(e.get("args", {}).get("bytes", 0) for e in events if "Memcpy HtoD" in e.get("name", ""))
    mask_bytes = 8 * 64 * 64 * 4
    assert copied < mask_bytes // 4, copied


def _shape_literal_cases() -> dict:
    from kornia.filters import get_gaussian_kernel1d
    from kornia.geometry.transform.affwarp import _compute_tensor_center, _compute_tensor_center3d

    def flip(cls):
        return lambda x: cls(p=1.0).compute_transformation(x, {}, {})

    return {
        "tensor_center": lambda x: _compute_tensor_center(x[:, :, 0]),
        "tensor_center3d": _compute_tensor_center3d,
        "gaussian_kernel1d": lambda x: get_gaussian_kernel1d(x.shape[-1], torch.full((1, 1), 1.5, device=x.device)),
        "RandomHorizontalFlip3D": flip(K.RandomHorizontalFlip3D),
        "RandomVerticalFlip3D": flip(K.RandomVerticalFlip3D),
        "RandomDepthicalFlip3D": flip(K.RandomDepthicalFlip3D),
    }


@pytest.mark.parametrize("name", sorted(_shape_literal_cases()))
def test_dynamo_shape_derived_literals_compile_once(name, device, torch_optimizer, optimizer_backend) -> None:
    # Literals built from sizes stay symbolic under a dynamic compile: one graph for every size, no guard on a value.
    from torch._dynamo.testing import CompileCounter

    if optimizer_backend == "jit":
        pytest.skip("counts the graphs of torch.compile, which torch.jit.script does not build")
    op = _shape_literal_cases()[name]
    counter = CompileCounter()
    compiled = torch_optimizer(op, backend=counter, dynamic=True, fullgraph=True)
    for size in (9, 11, 13):
        x = torch.rand(2, 3, size, size + 2, size + 4, device=device)
        assert torch.equal(compiled(x), op(x))
    assert counter.frame_count == 1
