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

from typing import Optional

import pytest
import torch
from torch import nn

import kornia
from kornia.contrib.edge_detection import EdgeDetector, EdgeDetectorBuilder
from kornia.core._compat import torch_version_ge, torch_version_lt
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
