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
# Learnable ranges made on CUDA for an augmentation that stays on the CPU: ``PlainUniformGenerator`` samples on the CPU
# and returns the draws on the device of the ranges.
_RANGES_ON_CUDA = {
    "RandomBrightness-range": lambda r: K.RandomBrightness(r(0.8, 1.2), p=1.0),
    "RandomBrightness-half-width": lambda r: K.RandomBrightness(r(0.2), p=1.0),
    "RandomHue-half-width": lambda r: K.RandomHue(r(0.1), p=1.0),
    "RandomRotation": lambda r: K.RandomRotation(r(-10.0, 10.0), p=1.0),
    "RandomGamma": lambda r: K.RandomGamma(r(0.8, 1.2), r(0.9, 1.1), p=1.0),
}
# Learnable ranges that move with the module, next to numbers that do not: the samplers of all of them are built again
# before each draw.
_LEARNABLE_FOLLOWED = {
    "RandomBrightness-half-width": lambda r: K.RandomBrightness(r(0.2), p=1.0),
    "RandomRGBShift-half-width": lambda r: K.RandomRGBShift(r(0.1), (-0.1, 0.1), 0.1, p=1.0),
    "RandomRotation-half-width": lambda r: K.RandomRotation(r(10.0), p=1.0),
    "RandomAffine-half-width": lambda r: K.RandomAffine(r(10.0), (0.1, 0.1), (0.9, 1.1), 5.0, p=1.0),
    "ColorJitter-half-width": lambda r: K.ColorJitter(r(0.2), 0.1, (0.9, 1.1), 0.1, p=1.0),
    "RandomPerspective": lambda r: K.RandomPerspective(r(0.3), p=1.0),
    "RandomMixUpV2": lambda r: K.RandomMixUpV2(r(0.0, 1.0), p=1.0),
    "RandomTranslate": lambda r: K.RandomTranslate(r(-0.1, 0.1), (-0.1, 0.1), p=1.0),
    "RandomMotionBlur": lambda r: K.RandomMotionBlur(3, r(30.0), (-0.5, 0.5), p=1.0),
    "RandomRotation3D": lambda r: K.RandomRotation3D(r(10.0), p=1.0),
    "RandomAffine3D": lambda r: K.RandomAffine3D(r(10.0), (0.1, 0.1, 0.1), (0.9, 1.1), 5.0, p=1.0),
    "RandomPerspective3D": lambda r: K.RandomPerspective3D(r(0.5), p=1.0),
}
# Auto operations and policies moved to CUDA as a whole (#6297). The range they read from their generator is on CUDA,
# where reading its values on the host would wait for the GPU.
_MOVED_AUTO = {
    "Brightness": lambda: KAuto.operations.ops.Brightness(initial_probability=1.0),
    "Posterize": lambda: KAuto.operations.ops.Posterize(initial_probability=1.0),
    "ShearX": lambda: KAuto.operations.ops.ShearX(initial_probability=1.0),
    "TranslateX": lambda: KAuto.operations.ops.TranslateX(initial_probability=1.0),
    "RandAugment-rotate": lambda: KAuto.RandAugment(n=1, m=10, policy=[[("rotate", -30.0, 30.0)]]),
    "RandAugment-shear_x": lambda: KAuto.RandAugment(n=1, m=10, policy=[[("shear_x", -0.3, 0.3)]]),
    "RandAugment-posterize": lambda: KAuto.RandAugment(n=1, m=10, policy=[[("posterize", 4.0, 8.0)]]),
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


@pytest.mark.applies_to(devices=("cuda",), reason="CUDA host synchronizations need a CUDA device")
class TestAugmentationHostSync(BaseTester):
    @pytest.mark.parametrize("case, which", _CALLS)
    def test_no_avoidable_host_sync(self, case, which, device, dtype) -> None:
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
        op = _EMPTY_BATCH_MIX[name]()
        x = torch.rand(0, 3, 32, 32, device=device, dtype=dtype)
        op(x)
        torch.cuda.synchronize()
        outputs = []
        sites = sync_sites(lambda: outputs.append(op(x)))
        assert outputs[0].shape == x.shape
        assert outputs[0].device == x.device
        assert not set(sites) - INHERENT_SITES.keys(), dict(sites)

    @pytest.mark.parametrize("name", sorted(_RANGES_ON_CUDA))
    def test_learnable_range_on_cuda_with_the_module_on_the_cpu(self, name, device, dtype) -> None:
        def learnable(*values):
            value = values if len(values) > 1 else values[0]
            return torch.nn.Parameter(torch.tensor(value, device=device, dtype=dtype))

        op = _RANGES_ON_CUDA[name](learnable)
        x = torch.rand(4, 3, 16, 16, device=device, dtype=dtype)
        op(x)
        torch.cuda.synchronize()
        outputs = []
        sites = sync_sites(lambda: outputs.append(op(x)))
        assert not set(sites) - INHERENT_SITES.keys(), dict(sites)
        outputs[0].sum().backward()
        assert all(p.grad is not None for p in op.parameters())

    @pytest.mark.parametrize("name", sorted(_LEARNABLE_FOLLOWED))
    def test_learnable_range_rebuilt_before_each_draw_adds_no_sync(self, name, device, dtype) -> None:
        # The samplers are built again before each draw (#6572); neither the range nor the numbers next to it are read.

        def learnable(*values):
            value = values if len(values) > 1 else values[0]
            return torch.nn.Parameter(torch.tensor(value, device=device, dtype=dtype))

        op = _LEARNABLE_FOLLOWED[name](learnable).to(device=device, dtype=dtype)
        volumetric = name.endswith("3D")
        x = (
            torch.rand(4, 3, 4, 16, 16, device=device, dtype=dtype)
            if volumetric
            else torch.rand(4, 3, 16, 16, device=device, dtype=dtype)
        )
        op(x)
        torch.cuda.synchronize()
        sites = sync_sites(lambda: op(x))
        # Compared with the same call without the rebuild, so that no site below ``make_samplers`` can hide from this.
        op._param_generator._forward_pre_hooks.clear()
        torch.cuda.synchronize()
        without = sync_sites(lambda: op(x))
        assert not sites - without, (dict(sites), dict(without))

    @pytest.mark.parametrize("name", sorted(_MOVED_AUTO))
    def test_moved_auto_operation_adds_no_sync_of_its_own(self, name, device, dtype) -> None:
        module = _MOVED_AUTO[name]().to(device=device, dtype=dtype)
        x = torch.rand(4, 3, 32, 32, device=device, dtype=dtype)
        module(x)
        torch.cuda.synchronize()
        sites = sync_sites(lambda: module(x))
        # The generators return their draws on the CPU (see the conventions page), which is not the wrapper's doing.
        assert not [site for site in sites if site.startswith("augmentation/auto/")], dict(sites)

    @pytest.mark.parametrize("same_on_batch", [False, True])
    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("name", sorted(_GATED))
    def test_gate_selection_unchanged(self, name, p, same_on_batch, device, dtype) -> None:
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


@pytest.mark.uses_accelerator
@pytest.mark.applies_to(devices=("cuda",), reason="CUDA host synchronizations need a CUDA device")
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: K.RandomHorizontalFlip(p=1.0), id="flip"),
        pytest.param(lambda: K.RandomAffine(15.0, p=0.5), id="affine_half"),
        pytest.param(lambda: K.RandomMosaic(output_size=(16, 16), p=1.0), id="mosaic"),
        pytest.param(lambda: K.RandomMosaic(output_size=(16, 16), p=0.5), id="mosaic_half"),
    ],
)
def test_list_boxes_add_no_host_sync_to_dense_boxes(build) -> None:
    # The padding mask of a list input lives on the host, so following it and trimming the export read nothing from the
    # device: a list costs what a tensor costs, apart from validating each of its tensors on its own.
    device = torch.device("cuda")
    x = torch.rand(4, 3, 16, 16, device=device)
    first = torch.tensor([[1.0, 1.0, 5.0, 5.0]], device=device)
    second = torch.tensor([[2.0, 2.0, 6.0, 7.0], [3.0, 1.0, 8.0, 4.0]], device=device)
    aug = K.AugmentationSequential(build(), data_keys=["input", "bbox_xyxy"])

    def sites(boxes) -> dict[str, int]:
        def call() -> None:
            # Several draws, so that a gate of p=0.5 runs its skipped, selected and partial paths.
            torch.manual_seed(0)
            for _ in range(8):
                aug(x, boxes)

        call()
        torch.cuda.synchronize()
        counted = sync_sites(call)
        del counted["geometry/boxes.py:_validate_box_values"]
        return dict(counted)

    assert sites([first, second, second[:0], first]) == sites(second[:1].expand(4, -1, -1))


@pytest.mark.uses_accelerator
@pytest.mark.applies_to(devices=("cuda",), reason="host-to-device copies need a CUDA device")
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


@pytest.mark.applies_to(
    when=lambda p: p["optimizer_backend"] != "jit",
    reason="counts the graphs of torch.compile, which torch.jit.script does not build",
)
@pytest.mark.parametrize("name", sorted(_shape_literal_cases()))
def test_dynamo_shape_derived_literals_compile_once(name, device, torch_optimizer, optimizer_backend) -> None:
    # Literals built from sizes stay symbolic under a dynamic compile: one graph for every size, no guard on a value.
    from torch._dynamo.testing import CompileCounter

    op = _shape_literal_cases()[name]
    counter = CompileCounter()
    compiled = torch_optimizer(op, backend=counter, dynamic=True, fullgraph=True)
    for size in (9, 11, 13):
        x = torch.rand(2, 3, size, size + 2, size + 4, device=device)
        assert torch.equal(compiled(x), op(x))
    assert counter.frame_count == 1


@pytest.mark.applies_to(
    devices=("cuda",),
    when=lambda p: p["optimizer_backend"] == "inductor",
    reason="counts the CUDA kernels of an Inductor graph",
)
def test_dynamo_literals_stay_graph_constants(device, torch_optimizer, optimizer_backend) -> None:
    # Under torch.compile a literal stays a constant of the graph, which Inductor folds into its consumer: the
    # compiled grayscale is one kernel, as before. Filling the weights on the device would add a stack kernel per call.
    x = torch.rand(8, 3, 64, 64, device=device)
    compiled = torch_optimizer(K.RandomGrayscale(p=1.0))
    compiled(x)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        compiled(x)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert len(kernels) == 1, kernels
