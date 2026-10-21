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

import importlib
from typing import Optional

import pytest
import torch
from torch import nn

import kornia
from kornia.contrib.edge_detection import EdgeDetector, EdgeDetectorBuilder
from kornia.core._compat import torch_version_ge, torch_version_lt
from kornia.core.mixin.onnx import ONNXExportMixin
from kornia.models.dexined import DexiNed
from kornia.models.processors import ResizePostProcessor, ResizePreProcessor

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as _OrtFail
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument as _OrtInvalidArgument
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph as _OrtInvalidGraph
except ImportError:  # the ONNX tests skip without onnxruntime
    _OrtFail = _OrtInvalidArgument = _OrtInvalidGraph = RuntimeError

# The interpolation modes an exported ``ResizePostProcessor`` resizes with; "area" raises, see its own test.
_EXPORTABLE_MODES = ["nearest", "bilinear", "bicubic", "nearest-exact"]

# ``to_onnx`` without ``dynamo=``: the dynamo exporter from torch 2.9, the TorchScript exporter before it.
_EXPORTERS = [pytest.param({}, id="default"), pytest.param({"dynamo": False}, id="legacy")]


def _run_onnx(op, x):
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])
    return torch.from_numpy(session.run(None, {session.get_inputs()[0].name: x.numpy()})[0])


def _small_edge_detector(post_processor: Optional[nn.Module] = None) -> EdgeDetector:
    """A detector whose model is one convolution and a sigmoid, behind the builder's resize pre- and post-processor."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1), nn.Sigmoid())
    return EdgeDetector(model, ResizePreProcessor(16, 16), post_processor or ResizePostProcessor())


class _DoubledPostProcessor(ResizePostProcessor):
    """A ``ResizePostProcessor`` subclass with its own ``forward``: the maps at the model's size, doubled."""

    def forward(self, imgs, original_sizes):
        return 2 * imgs


class _NearestPostProcessor(ResizePostProcessor):
    """A ``ResizePostProcessor`` subclass that keeps its ``forward``: nearest resizing by default."""

    def __init__(self) -> None:
        super().__init__("nearest")


class _DelegatingPostProcessor(ResizePostProcessor):
    """A ``ResizePostProcessor`` subclass whose own ``forward`` delegates: a list eagerly, the maps under export."""

    def forward(self, imgs, original_sizes):
        return super().forward(imgs, original_sizes)


class _PerImageMaps(nn.Module):
    """A post-processor that returns a list with one ``(1, 1, h, w)`` map per image, at the model's size (#6161)."""

    def forward(self, imgs, original_sizes):
        return [imgs[i : i + 1] for i in range(imgs.shape[0])]


class _DrawingNormalisedModel(nn.Module):
    """A convolution behind a batch norm, which a forward pass in training mode would update, drawing in any mode."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x):
        return self.norm(self.conv(x + 0.0 * torch.rand_like(x)))


# One exporter per ID. conftest.py deselects IDs that contain "dynamo" unless KORNIA_TEST_OPTIMIZER is set.
_EACH_EXPORTER = [
    pytest.param({"dynamo": False}, id="torchscript"),
    pytest.param(
        {"dynamo": True},
        id="torch_export",
        marks=pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON),
    ),
]

# A caller's dynamic batch: the TorchScript exporter reads ``dynamic_axes``, the dynamo exporter ``dynamic_shapes``
# from torch 2.9, where ``to_onnx`` declares its own dynamic dimensions with it.
_DYNAMIC_BATCH = [
    pytest.param({"dynamo": False, "dynamic_axes": {"input": {0: "batch"}}}, id="torchscript_batch_override"),
    pytest.param(
        {"dynamo": True, "dynamic_shapes": ({0: "batch"},)},
        id="torch_export_batch_override",
        marks=[
            pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON),
            pytest.mark.skipif(torch_version_lt(2, 9, 0), reason="to_onnx uses dynamic_shapes from torch 2.9"),
        ],
    ),
]


def _count_probes(monkeypatch):
    """Count the runs that read the detector's output kind, wherever ``to_onnx`` or its check makes them."""
    mixin = importlib.import_module("kornia.core.mixin.onnx")
    probe = mixin._returns_list
    calls = []

    def _counting(module, inputs):
        calls.append(type(module).__name__)
        return probe(module, inputs)

    monkeypatch.setattr(kornia.contrib.edge_detection, "_returns_list", _counting)
    monkeypatch.setattr(mixin, "_returns_list", _counting)
    return calls


def _importorskip_exporter(export_kwargs):
    pytest.importorskip("onnx")
    if export_kwargs.get("dynamo", torch_version_ge(2, 9, 0)):
        pytest.importorskip("onnxscript")


class TestEdgeDetector(BaseTester):
    def test_builder_feeds_dexined_bgr_minus_bgr_mean(self, device, dtype):
        # DexiNed was trained on ``cv2.imread`` images (BGR, 0-255) minus ``mean_bgr``: xavysp/DexiNed@d944b70
        # datasets.py:255, 300, with ``--mean_pixel_values`` [103.939, 116.779, 123.68] in B, G, R order
        # (main.py:319-320).
        # A 64 px network input keeps the random DexiNed fast in half precision on the CPU.
        det = EdgeDetectorBuilder.build(pretrained=False, image_size=64).to(device, dtype)
        # DexiNed stays at ``model[1]``, so the state-dict keys of a saved ``EdgeDetector`` stay ``model.1.*``.
        assert isinstance(det.model[1], DexiNed)
        dexined = next(m for m in det.model.modules() if isinstance(m, DexiNed))

        captured = {}

        def capture(name):
            def hook(module, args):
                captured[name] = args[0].clone()

            return hook

        handles = [
            det.model.register_forward_pre_hook(capture("seq_in")),
            dexined.register_forward_pre_hook(capture("dexined_in")),
        ]
        try:
            det(torch.rand(1, 3, 64, 96, device=device, dtype=dtype))
        finally:
            for handle in handles:
                handle.remove()

        # Compared in float64 on the CPU (MPS has no float64).
        seq_in = captured["seq_in"].cpu().double()  # the resized RGB image in [0, 1]
        mean_bgr = torch.tensor([103.939, 116.779, 123.68], dtype=torch.float64).view(1, 3, 1, 1)
        expected = 255 * seq_in.flip(1) - mean_bgr
        # Normalize computes ``(x - m / 255) / (1 / 255)`` in ``dtype`` from float32 statistics: about four roundings,
        # each up to half an ulp of the coarser of ``dtype`` and float32 on the 0-255 scale, so 4 * 255 * eps bounds
        # the error (bfloat16: 8.0, below the 19.7 between the R and B means that a mean in the wrong order leaves).
        atol = 4 * 255 * max(torch.finfo(dtype).eps, torch.finfo(torch.float32).eps)
        self.assert_close(captured["dexined_in"].cpu().double(), expected, rtol=0.0, atol=atol)

    @pytest.mark.device_agnostic
    def test_edge_detector_onnx_matches_eager(self):
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxscript")

        det = EdgeDetectorBuilder.build(pretrained=False)
        # ``image_size=None`` exports ``det.model`` (the BGR step, DexiNed and the sigmoid) with a dynamic batch,
        # height and width.
        op = det.to_onnx(image_size=None, include_pre_and_post_processor=False, save=False)
        session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])

        img = torch.rand(2, 3, 64, 96)
        exported = torch.from_numpy(session.run(None, {session.get_inputs()[0].name: img.numpy()})[0])
        with torch.no_grad():
            eager = det.model(img)
        self.assert_close(exported, eager, rtol=0.0, atol=1e-4)

    def test_list_of_two_sizes_returns_a_map_per_image_at_its_size(self, device, dtype):
        det = EdgeDetectorBuilder.build(pretrained=False, image_size=32).to(device, dtype)
        generator = torch.Generator().manual_seed(0)
        images = [torch.rand(3, h, w, generator=generator).to(device, dtype) for h, w in [(40, 72), (33, 47)]]

        out = det(images)

        assert isinstance(out, list)
        assert [tuple(edge_map.shape) for edge_map in out] == [(1, 1, 40, 72), (1, 1, 33, 47)]
        for image, edge_map in zip(images, out):
            self.assert_close(edge_map, det(image[None])[0])

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EXPORTERS)
    @pytest.mark.parametrize("interpolation", _EXPORTABLE_MODES)
    @pytest.mark.parametrize(
        ("image_size", "shapes"),
        [
            # different heights and widths, an odd size, sizes above and below the pre-processor's 32
            (None, [(1, 3, 40, 72), (2, 3, 33, 47), (3, 3, 64, 48), (1, 3, 20, 28)]),
            # a graph input size other than the pre-processor's, which keeps its own
            (48, [(1, 3, 48, 48), (2, 3, 48, 48), (3, 3, 48, 48)]),
        ],
        ids=["dynamic_size", "fixed_size"],
    )
    def test_onnx_with_pre_and_post_processor_matches_eager(self, export_kwargs, interpolation, image_size, shapes):
        # The graph returns one (B, 1, H, W) tensor at the input's size, resized with the post-processor's mode:
        # ``torch.cat`` of the eager per-image list.
        pytest.importorskip("onnx")
        if export_kwargs.get("dynamo", torch_version_ge(2, 9, 0)):
            pytest.importorskip("onnxscript")
        torch.manual_seed(0)
        det = EdgeDetectorBuilder.build(pretrained=False, image_size=32)
        det.post_processor = ResizePostProcessor(interpolation)
        op = det.to_onnx(image_size=image_size, save=False, **export_kwargs)

        generator = torch.Generator().manual_seed(0)
        for shape in shapes:
            images = torch.rand(*shape, generator=generator)
            with torch.no_grad():
                expected = torch.cat(det(images))
            assert expected.shape == (shape[0], 1, *shape[2:])
            self.assert_close(_run_onnx(op, images), expected, rtol=0.0, atol=1e-4)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize(
        "image_size",
        [
            pytest.param(
                None,
                id="dynamic_size",
                marks=[
                    pytest.mark.xfail(
                        torch_version_lt(2, 6, 0),
                        reason=(
                            "the dynamo exporter of torch 2.5.1 writes a resize to a symbolic size, as "
                            "F.interpolate(x, size=y.shape[-2:]) alone, as a Concat of 0-d sizes, which ONNX Runtime "
                            "rejects as INVALID_GRAPH"
                        ),
                        raises=_OrtInvalidGraph,
                        strict=True,
                    ),
                    pytest.mark.xfail(
                        torch_version_ge(2, 6, 0) and torch_version_lt(2, 9, 0),
                        reason=(
                            "not checked on torch 2.6 to 2.8, whose dynamo exporter may write a resize to a symbolic "
                            "size as torch 2.5.1's does, which ONNX Runtime rejects"
                        ),
                        raises=(_OrtInvalidGraph, _OrtFail, _OrtInvalidArgument),
                        strict=False,
                    ),
                ],
            ),
            pytest.param(24, id="fixed_size"),
        ],
    )
    def test_onnx_on_the_torch_export_exporter_returns_the_input_size(self, image_size):
        # Below torch 2.9, ``dynamo=True`` selects the exporter on which ``torch.onnx.is_in_onnx_export()`` is False,
        # so only the flag that ``to_onnx`` sets selects the export branch there. A one-convolution model: that
        # exporter cannot translate DexiNed's batch norm on torch 2.5.1.
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        det = _small_edge_detector()
        op = det.to_onnx(image_size=image_size, save=False, dynamo=True)

        generator = torch.Generator().manual_seed(0)
        shapes = [(1, 3, 20, 28), (3, 3, 33, 18)] if image_size is None else [(1, 3, 24, 24), (3, 3, 24, 24)]
        for shape in shapes:
            images = torch.rand(*shape, generator=generator)
            with torch.no_grad():
                expected = torch.cat(det(images))
            self.assert_close(_run_onnx(op, images), expected, rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "dynamo",
        [
            pytest.param(False, id="legacy"),
            pytest.param(
                True,
                id="torch_export",
                marks=pytest.mark.skipif(
                    torch_version_lt(2, 9, 0),
                    reason="below torch 2.9 a direct export on that exporter takes the eager path; use to_onnx",
                ),
            ),
        ],
    )
    def test_direct_torch_onnx_export_matches_eager(self, tmp_path, dynamo):
        # ``torch.onnx.export(detector, ...)`` without ``to_onnx``: ``torch.onnx.is_in_onnx_export()`` alone selects
        # the export branch, on the legacy exporter and on the dynamo exporter from torch 2.9.
        pytest.importorskip("onnx")
        if dynamo:
            pytest.importorskip("onnxscript")
        det = _small_edge_detector().eval()
        path = tmp_path / "edge_detector.onnx"
        traced = torch.rand(2, 3, 24, 32, generator=torch.Generator().manual_seed(1))
        if dynamo:
            dims = {0: "batch", 2: "height", 3: "width"}
            torch.onnx.export(det, (traced,), str(path), dynamo=True, input_names=["input"], dynamic_shapes=(dims,))
        else:
            axes = {0: "batch", 2: "height", 3: "width"}
            torch.onnx.export(
                det,
                (traced,),
                str(path),
                dynamo=False,
                opset_version=18,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": axes, "output": axes},
            )
        onnx = pytest.importorskip("onnx")
        op = onnx.load(str(path))

        generator = torch.Generator().manual_seed(0)
        for shape in [(1, 3, 20, 28), (3, 3, 33, 18)]:
            images = torch.rand(*shape, generator=generator)
            with torch.no_grad():
                expected = torch.cat(det(images))
            self.assert_close(_run_onnx(op, images), expected, rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EXPORTERS)
    def test_onnx_with_an_area_post_processor_raises(self, export_kwargs):
        # "area" resizing is adaptive average pooling, which cannot be exported to a size known only at run time:
        # the legacy exporter fails on it and the dynamo exporter's graph is right only at the traced size.
        pytest.importorskip("onnx")
        if export_kwargs.get("dynamo", torch_version_ge(2, 9, 0)):
            pytest.importorskip("onnxscript")
        det = _small_edge_detector(ResizePostProcessor("area"))
        images = torch.rand(2, 3, 20, 28, generator=torch.Generator().manual_seed(0))

        with pytest.raises(ValueError, match='"area"'):
            det.to_onnx(image_size=None, save=False, **export_kwargs)
        with pytest.raises(ValueError, match='"area"'):
            det.to_onnx(image_size=24, save=False, **export_kwargs)

        # eager resizing with "area" is unchanged
        out = det(images)
        assert isinstance(out, list)
        assert [tuple(edge_map.shape) for edge_map in out] == [(1, 1, 20, 28)] * 2
        assert det._exporting_to_onnx is False

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EXPORTERS)
    def test_onnx_keeps_the_forward_of_a_post_processor_subclass(self, export_kwargs):
        # Only ``ResizePostProcessor.forward`` itself is replaced by the export branch: a subclass that overrides it is
        # exported as its own ``forward`` computes.
        pytest.importorskip("onnx")
        if export_kwargs.get("dynamo", torch_version_ge(2, 9, 0)):
            pytest.importorskip("onnxscript")
        det = _small_edge_detector(_DoubledPostProcessor())
        op = det.to_onnx(image_size=None, save=False, **export_kwargs)

        generator = torch.Generator().manual_seed(0)
        for shape in [(1, 3, 20, 28), (3, 3, 33, 18)]:
            images = torch.rand(*shape, generator=generator)
            with torch.no_grad():
                expected = det(images)
            assert expected.shape == (shape[0], 1, 16, 16)
            self.assert_close(_run_onnx(op, images), expected, rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EXPORTERS)
    def test_onnx_resizes_back_for_a_post_processor_subclass_that_keeps_forward(self, export_kwargs):
        # A subclass without its own ``forward`` resizes as ``ResizePostProcessor`` does, so the export branch replaces
        # it too, with the subclass's mode.
        pytest.importorskip("onnx")
        if export_kwargs.get("dynamo", torch_version_ge(2, 9, 0)):
            pytest.importorskip("onnxscript")
        det = _small_edge_detector(_NearestPostProcessor())
        op = det.to_onnx(image_size=None, save=False, **export_kwargs)

        generator = torch.Generator().manual_seed(0)
        for shape in [(1, 3, 20, 28), (3, 3, 33, 18)]:
            images = torch.rand(*shape, generator=generator)
            with torch.no_grad():
                expected = torch.cat(det(images))
            self.assert_close(_run_onnx(op, images), expected, rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "dynamo",
        [
            pytest.param(False, id="legacy"),
            pytest.param(
                True,
                id="torch_export",
                marks=pytest.mark.skipif(
                    torch_version_lt(2, 9, 0),
                    reason="below torch 2.9 a direct export on that exporter takes the eager path; use to_onnx",
                ),
            ),
        ],
    )
    def test_direct_torch_onnx_export_with_an_area_post_processor_raises(self, tmp_path, dynamo):
        # ``forward`` itself rejects "area" under export: the dynamo exporter would otherwise give a graph that is
        # right only at the traced size. That exporter raises its own error, caused by the ``ValueError``.
        pytest.importorskip("onnx")
        if dynamo:
            pytest.importorskip("onnxscript")
        det = _small_edge_detector(ResizePostProcessor("area")).eval()
        traced = torch.rand(2, 3, 24, 32, generator=torch.Generator().manual_seed(1))
        dims = {0: "batch", 2: "height", 3: "width"}
        export_kwargs = (
            {"dynamic_shapes": (dims,)}
            if dynamo
            else {"opset_version": 18, "output_names": ["output"], "dynamic_axes": {"input": dims, "output": dims}}
        )

        # torch 2.9's non-strict export registers a global buffer-registration hook and removes it only when the
        # trace succeeds. Drop the one this failing export leaves behind, or every later dynamo trace in the process
        # calls it and raises ``Unsupported``.
        buffer_hooks = torch.nn.modules.module._global_buffer_registration_hooks
        hooks_before = set(buffer_hooks)
        try:
            with pytest.raises(Exception) as info:
                torch.onnx.export(
                    det,
                    (traced,),
                    str(tmp_path / "edge_detector.onnx"),
                    dynamo=dynamo,
                    input_names=["input"],
                    **export_kwargs,
                )
        finally:
            for key in set(buffer_hooks) - hooks_before:
                del buffer_hooks[key]

        error = info.value
        while error is not None and not isinstance(error, ValueError):
            error = error.__cause__ or error.__context__
        assert error is not None, f"no ValueError behind {info.value!r}"
        assert '"area"' in str(error)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EACH_EXPORTER)
    def test_onnx_with_a_per_image_post_processor_exports_one_image_6161(self, export_kwargs):
        # The exporters turn a list into one graph output per traced image. Exported for a dynamic batch, the
        # TorchScript graph returned the traced batch's two maps at any batch size, and the dynamo exporter failed;
        # such a detector is exported for one image, a batch that ONNX Runtime enforces.
        _importorskip_exporter(export_kwargs)
        ort = pytest.importorskip("onnxruntime")
        det = _small_edge_detector(_PerImageMaps())
        images = torch.rand(3, 3, 24, 24, generator=torch.Generator().manual_seed(0))
        assert len(det(images)) == 3

        op = det.to_onnx(image_size=24, save=False, **export_kwargs)

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
        assert op.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 1
        session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"input": images[:1].numpy()})
        assert len(outputs) == 1
        with torch.no_grad():
            expected = det(images[:1])
        self.assert_close(torch.from_numpy(outputs[0]), expected[0], rtol=0.0, atol=1e-5)
        with pytest.raises(_OrtInvalidArgument):
            session.run(None, {"input": images.numpy()})
        assert det._exporting_to_onnx is False

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("override", _DYNAMIC_BATCH)
    def test_onnx_with_a_per_image_post_processor_rejects_a_dynamic_batch_6161(self, override, tmp_path):
        # A caller's dynamic batch would give back the traced batch's maps at any batch size: the export raises after
        # it runs, before anything is saved, and names the cause.
        _importorskip_exporter(override)
        det = _small_edge_detector(_PerImageMaps())
        path = tmp_path / "edges.onnx"

        with pytest.raises(ValueError, match="EdgeDetector returns one output per image") as info:
            det.to_onnx(onnx_name=str(path), image_size=24, **override)

        assert "ResizePostProcessor" in str(info.value)
        assert not path.exists()
        assert det._exporting_to_onnx is False

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    def test_onnx_with_a_batch_dependent_post_processor_rejects_a_dynamic_export_6197(self, tmp_path):
        # A list only for more than one image: the one-image reading is a tensor, so the export is dynamic, and the
        # check after it reads the traced batch of two.
        pytest.importorskip("onnx")

        class _ListBeyondOneImage(ResizePostProcessor):
            def forward(self, imgs, original_sizes):
                return imgs if imgs.shape[0] == 1 else list(imgs.split(1))

        det = _small_edge_detector(_ListBeyondOneImage())
        path = tmp_path / "edges.onnx"

        with pytest.raises(ValueError, match="EdgeDetector returns one output per image"):
            det.to_onnx(onnx_name=str(path), image_size=24, dynamo=False)

        assert not path.exists()

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("change", ["forward", "hook", "global_hook"])
    def test_onnx_of_a_detector_whose_own_output_is_a_list_exports_one_image_6197(self, change):
        # A ResizePostProcessor alone skips the probe, but a detector whose forward, or a hook on it, turns the maps
        # into a per-image list is probed, and exported for one image.
        pytest.importorskip("onnx")

        class _ListDetector(EdgeDetector):
            def forward(self, images):
                return list(super().forward(images))

        if change == "forward":
            small = _small_edge_detector()
            det = _ListDetector(small.model, small.pre_processor, small.post_processor)
        elif change == "hook":
            det = _small_edge_detector()
            det.register_forward_hook(lambda module, inputs, output: list(output))
        else:
            det = _small_edge_detector()

            def _listed(module, inputs, output):
                return list(output) if module is det else output

            handle = nn.modules.module.register_module_forward_hook(_listed)
        try:
            assert not det._returns_one_tensor_on_export()
            op = det.to_onnx(image_size=24, save=False, dynamo=False)
        finally:
            if change == "global_hook":
                handle.remove()

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
        assert len(op.graph.output) == 1

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", [*_EACH_EXPORTER, *_DYNAMIC_BATCH])
    def test_onnx_with_a_resize_post_processor_keeps_a_dynamic_batch_without_a_probe_6161(
        self, export_kwargs, monkeypatch
    ):
        # The export replaces a ResizePostProcessor with one resize of the batch, so its output is one tensor: no
        # forward pass reads the output kind, before the export or after it, and a caller's dynamic batch exports too.
        _importorskip_exporter(export_kwargs)

        def _no_probe(module, inputs):
            raise AssertionError("to_onnx ran the detector to read its output kind")

        monkeypatch.setattr(kornia.contrib.edge_detection, "_returns_list", _no_probe)
        monkeypatch.setattr(importlib.import_module("kornia.core.mixin.onnx"), "_returns_list", _no_probe)
        det = _small_edge_detector()

        op = det.to_onnx(image_size=24, save=False, **export_kwargs)

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        generator = torch.Generator().manual_seed(0)
        for batch in (1, 3):
            images = torch.rand(batch, 3, 24, 24, generator=generator)
            with torch.no_grad():
                expected = torch.cat(det(images))
            self.assert_close(_run_onnx(op, images), expected, rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", [*_EACH_EXPORTER, *_DYNAMIC_BATCH])
    def test_onnx_with_a_tensor_post_processor_keeps_a_dynamic_batch_6161(self, export_kwargs):
        # A post-processor of the caller's own that returns one tensor for the batch is read as such: the batch stays
        # dynamic, also through a caller's dynamic batch.
        _importorskip_exporter(export_kwargs)
        det = _small_edge_detector(_DoubledPostProcessor())

        op = det.to_onnx(image_size=24, save=False, **export_kwargs)

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        images = torch.rand(3, 3, 24, 24, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            expected = det(images)
        self.assert_close(_run_onnx(op, images), expected, rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EACH_EXPORTER)
    def test_onnx_model_only_of_a_per_image_detector_keeps_a_dynamic_batch_6161(self, export_kwargs):
        # Without the pre- and post-processor, the model's tensor output follows any batch, whatever the
        # post-processor returns.
        _importorskip_exporter(export_kwargs)
        det = _small_edge_detector(_PerImageMaps())

        op = det.to_onnx(image_size=24, include_pre_and_post_processor=False, save=False, **export_kwargs)

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        images = torch.rand(3, 3, 24, 24, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            expected = det.model(images)
        self.assert_close(_run_onnx(op, images), expected, rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EACH_EXPORTER)
    def test_onnx_with_a_delegating_resize_post_processor_exports_one_image_6161(self, export_kwargs):
        # A subclass whose own ``forward`` calls ``ResizePostProcessor.forward`` returns a list eagerly and the maps
        # unchanged under export. The output kind is read outside the export, so it is exported for one image.
        _importorskip_exporter(export_kwargs)
        det = _small_edge_detector(_DelegatingPostProcessor())

        op = det.to_onnx(image_size=16, save=False, **export_kwargs)

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
        images = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            expected = det(images)
        self.assert_close(_run_onnx(op, images), expected[0], rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        ("post_processor", "override"),
        [
            pytest.param(_DoubledPostProcessor, {"dynamo": False}, id="tensor_torchscript"),
            pytest.param(_PerImageMaps, {"dynamo": False}, id="list_torchscript"),
            pytest.param(
                _PerImageMaps,
                {"dynamo": False, "dynamic_axes": {"input": {0: "batch"}}},
                id="list_torchscript_batch_override",
            ),
        ],
    )
    def test_onnx_rechecks_tensor_output_at_the_tracing_batch_6161(self, post_processor, override, monkeypatch):
        # A list result already requires a fixed batch. A tensor result is checked again on the tracing batch.
        # Neither reading is kept after the export.
        _importorskip_exporter(override)
        calls = _count_probes(monkeypatch)
        det = _small_edge_detector(post_processor())

        if "dynamic_axes" in override:
            with pytest.raises(ValueError, match="returns one output per image"):
                det.to_onnx(image_size=24, save=False, **override)
        else:
            det.to_onnx(image_size=24, save=False, **override)

        assert calls == ["EdgeDetector"] * (2 if post_processor is _DoubledPostProcessor else 1)
        assert "_onnx_list_output" not in det.__dict__

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    def test_direct_mixin_export_of_a_per_image_detector_reads_the_output_kind_6161(self, monkeypatch):
        # ``ONNXExportMixin.to_onnx`` called directly reads nothing before the export, so the check after it runs the
        # detector itself, and rejects the dynamic batch.
        pytest.importorskip("onnx")
        calls = _count_probes(monkeypatch)
        det = _small_edge_detector(_PerImageMaps())

        with pytest.raises(ValueError, match="EdgeDetector returns one output per image"):
            ONNXExportMixin.to_onnx(
                det, input_shape=[-1, 3, 24, 24], pseudo_shape=[1, 3, 24, 24], save=False, dynamo=False
            )

        assert calls == ["EdgeDetector"]

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    def test_onnx_output_probe_leaves_the_detector_as_it_was_6161(self):
        # The run that reads the output kind is in eval mode, on the model's dtype, without gradients: the batch norm
        # keeps its statistics, each submodule its training flag, and the global random state is given back.
        pytest.importorskip("onnx")

        def _draws_after(export):
            torch.manual_seed(0)
            model = _DrawingNormalisedModel()
            det = EdgeDetector(model, ResizePreProcessor(16, 16), _PerImageMaps()).double().train()
            det.pre_processor.eval()  # a mixed state, which ``train()`` would not restore
            running_mean = model.norm.running_mean.clone()
            if export:
                op = det.to_onnx(image_size=24, save=False, dynamo=False)
                assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
                assert torch.equal(model.norm.running_mean, running_mean)
                assert model.norm.num_batches_tracked.item() == 0
                assert det.training
                assert model.training
                assert model.norm.training
                assert not det.pre_processor.training
            return torch.rand(8)

        self.assert_close(_draws_after(True), _draws_after(False))

    def test_dynamo_returns_the_eager_list(self, device, dtype, torch_optimizer):
        # Only an export returns the stacked tensor: a compiled detector keeps the per-image list. Under
        # ``torch.compile`` on torch 2.5.1, ``is_exporting()`` is True (#5037), so it cannot select the export branch.
        det = _small_edge_detector().to(device, dtype)
        images = torch.rand(2, 3, 20, 28, generator=torch.Generator().manual_seed(0)).to(device, dtype)

        out = torch_optimizer(det)(images)

        assert isinstance(out, list)
        expected = det(images)
        assert len(out) == len(expected) == 2
        for actual, edge_map in zip(out, expected):
            self.assert_close(actual, edge_map)

    @pytest.mark.slow
    def test_smoke(self, device, dtype):
        img = torch.rand(2, 3, 64, 64, device=device, dtype=dtype)
        net = kornia.contrib.EdgeDetectorBuilder.build(pretrained=False).to(device, dtype)
        out = net(img)
        # ResizePostProcessor returns a list, so we need to handle that
        if isinstance(out, list):
            assert len(out) == 2
            assert all(item.shape == (1, 1, 64, 64) for item in out)
        else:
            assert out.shape == (2, 1, 64, 64)

    @pytest.mark.slow
    @pytest.mark.skip(reason="issue with `ClassVar[list[int]]`")
    def test_jit(self, device, dtype):
        op = kornia.contrib.EdgeDetectorBuilder.build(pretrained=False).to(device, dtype)
        op_jit = torch.jit.script(op)
        assert op_jit is not None
