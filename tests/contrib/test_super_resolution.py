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
import re
import sys
from typing import Optional

import pytest
import torch
from torch import nn

import kornia
from kornia.contrib.super_resolution import (
    RRDBNetBuilder,
    SmallSRBuilder,
    SuperResolution,
    SuperResolutionConfig,
)
from kornia.core._compat import torch_version_ge, torch_version_lt
from kornia.models.processors import OutputRangePostProcessor, ResizePreProcessor

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


class TestSuperResolutionBuilders(BaseTester):
    """Pin the public super-resolution entry points.

    Regression test for #4291: ``SuperResolution`` did not implement ``ModelBase``'s
    abstract ``from_config`` and defined no ``__init__``, so every builder raised at
    construction. Nothing in ``tests/`` called the builders, which is why CI never
    noticed.
    """

    def test_small_sr_builder_builds_and_runs(self, device, dtype):
        model = SmallSRBuilder.build(pretrained=False).to(device, dtype)
        out = model(torch.rand(1, 3, 16, 16, device=device, dtype=dtype))
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    def test_rrdbnet_builder_builds_and_runs(self, device, dtype):
        model = RRDBNetBuilder.build(pretrained=False).to(device, dtype)
        out = model(torch.rand(1, 3, 16, 16, device=device, dtype=dtype))
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    @pytest.mark.parametrize("num_images", [1, 2])
    def test_rrdb_list_input(self, device, dtype, num_images):
        # Every RRDB variant has the same identity pre-processor; the 6-block one keeps CPU half precision fast.
        model = RRDBNetBuilder.build("RealESRGAN_x4plus_anime_6B", pretrained=False).to(device, dtype)
        images = [torch.rand(3, 16, 20, device=device, dtype=dtype) for _ in range(num_images)]

        out = model(images)

        assert isinstance(out, list)
        assert len(out) == num_images
        for image, actual in zip(images, out):
            # RealESRGAN_x4plus_anime_6B upscales by 4
            assert actual.shape == (3, 64, 80)
            self.assert_close(actual, model(image[None])[0])

    def test_save_pil_writes_the_torch_pixels(self, tmp_path):
        pytest.importorskip("PIL.Image")
        from kornia.io import ImageLoadType, load_image

        # the 6-block RRDB variant, random init, on small images
        model = RRDBNetBuilder.build("RealESRGAN_x4plus_anime_6B", pretrained=False)
        images = torch.rand(2, 3, 16, 20)

        written = {}
        for output_type in ("torch", "pil"):
            directory = tmp_path / output_type
            model.save(images, directory=str(directory), output_type=output_type)
            # file names carry a "_YYYYmmdd_HHMMSS" timestamp, which may differ between the two calls
            written[output_type] = {
                re.sub(r"_\d{8}_\d{6}", "", p.name): load_image(str(p), ImageLoadType.UNCHANGED)
                for p in sorted(directory.iterdir())
            }

        # source and super-resolved image of each of the two images, under the same names and with the same pixels
        assert len(written["torch"]) == 4
        assert written["pil"].keys() == written["torch"].keys()
        for name, decoded in written["torch"].items():
            assert torch.equal(written["pil"][name], decoded), name

    @pytest.mark.parametrize("model_name", ["small_sr", "RealESRNet_x4plus"])
    def test_from_config_dispatches_to_both_families(self, device, dtype, model_name):
        config = SuperResolutionConfig(model_name=model_name, pretrained=False)
        model = SuperResolution.from_config(config).to(device, dtype)
        out = model(torch.rand(1, 3, 16, 16, device=device, dtype=dtype))
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    def test_from_config_rejects_unknown_model_name(self):
        with pytest.raises(ValueError, match="small_sr"):
            SuperResolution.from_config(SuperResolutionConfig(model_name="not_a_model", pretrained=False))

    @pytest.mark.parametrize("builder", [SmallSRBuilder, RRDBNetBuilder])
    def test_onnx_metadata_is_initialised(self, builder):
        """``to_onnx`` reads these three; without defaults they are annotations only.

        Regression test for the review of #4335: an ``__init__`` that does not set them
        leaves export raising ``AttributeError``.
        """
        model = builder.build(pretrained=False)
        assert model.pseudo_image_size is None or isinstance(model.pseudo_image_size, int)
        assert model.input_image_size is None or isinstance(model.input_image_size, int)
        assert model.output_image_size is None or isinstance(model.output_image_size, int)

    def test_rrdbnet_onnx_metadata_defaults_to_none(self):
        """``RRDBNetBuilder`` sets none of the three, so they must default to ``None``."""
        model = RRDBNetBuilder.build(pretrained=False)
        assert model.input_image_size is None
        assert model.output_image_size is None
        assert model.pseudo_image_size is None

    # The exporter's cost scales with the traced node count, not with the pseudo input size, so
    # `RRDBNetBuilder`'s 23-block default costs ~4x the 6-block anime variant while traversing the
    # identical export path. Which variant maps to which architecture is pinned in
    # tests/models/test_rrdbnet.py::TestRRDBNetBuilder.
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        ("builder", "build_kwargs"),
        [
            (SmallSRBuilder, {}),
            pytest.param(
                RRDBNetBuilder,
                {"model_name": "RealESRGAN_x4plus_anime_6B"},
                # torch < 2.9 defaults ``torch.onnx.export`` to the TorchScript exporter, which walks this
                # graph in ~130 s on an Apple Silicon Mac and past the budget on the macOS runners; the
                # dynamo exporter (2.9+) takes ~6 s on the same machine.
                marks=pytest.mark.skipif(
                    sys.platform == "darwin" and torch_version_lt(2, 9, 0),
                    reason="TorchScript exporter (torch < 2.9) exceeds the 120-second timeout on macOS",
                ),
            ),
        ],
        ids=["SmallSRBuilder", "RRDBNetBuilder"],
    )
    def test_to_onnx_exports(self, builder, build_kwargs):
        """Both families export; this fails with ``AttributeError`` if the metadata is unset."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        builder.build(pretrained=False, **build_kwargs).to_onnx(save=False)

    @pytest.mark.device_agnostic
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", [{}, {"dynamo": False}], ids=["default", "legacy"])
    def test_to_onnx_with_pre_and_post_processor_matches_eager_at_any_size(self, export_kwargs):
        """Without an ``image_size``, the graph takes any batch, height and width, as the eager pre-processor does."""
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        if export_kwargs.get("dynamo", torch_version_ge(2, 9, 0)):
            pytest.importorskip("onnxscript")
        torch.manual_seed(0)
        model = SmallSRBuilder.build(pretrained=False)
        op = model.to_onnx(save=False, **export_kwargs)
        dims = [d.dim_param or d.dim_value for d in op.graph.input[0].type.tensor_type.shape.dim]
        assert dims[1] == 3
        assert all(isinstance(d, str) for d in (dims[0], dims[2], dims[3]))
        session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])

        generator = torch.Generator().manual_seed(0)
        # batches other than the traced 2, H != W, odd sizes
        for shape in [(1, 3, 40, 56), (3, 3, 37, 61)]:
            images = torch.rand(*shape, generator=generator)
            (upscaled,) = session.run(None, {session.get_inputs()[0].name: images.numpy()})
            with torch.no_grad():
                expected = model(images)
            self.assert_close(torch.from_numpy(upscaled), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(("upscale_factor", "image_size"), [(2, 32), (3, None), (4, 64)])
    def test_output_image_size_matches_the_real_forward(self, device, dtype, upscale_factor, image_size):
        """``output_image_size`` must equal what a forward actually returns.

        The pre-processor resizes every input to 224 before the network, so the output
        side is ``224 * upscale_factor`` and does not depend on ``image_size``. Pinning
        the attribute alone let it disagree with the model it describes.
        """
        model = SmallSRBuilder.build(pretrained=False, upscale_factor=upscale_factor, image_size=image_size).to(
            device, dtype
        )
        out = model(torch.rand(1, 3, 32, 32, device=device, dtype=dtype))
        assert out.shape[-1] == model.output_image_size
        assert out.shape[-2] == model.output_image_size
        assert model.output_image_size == 224 * upscale_factor

    def test_to_onnx_with_only_an_input_size_uses_it_for_the_dummy_input(self):
        """A fixed ``input_image_size`` without a ``pseudo_image_size`` is also the size of the dummy input."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        model = SuperResolution(torch.nn.Conv2d(3, 3, 3, padding=1), torch.nn.Identity(), torch.nn.Identity())
        model.input_image_size = 64
        exported = model.to_onnx(save=False, include_pre_and_post_processor=False)
        dims = exported.graph.input[0].type.tensor_type.shape.dim
        assert [d.dim_value for d in dims[1:]] == [3, 64, 64]

    def test_onnx_export_shape_matches_the_real_forward(self):
        """The static size handed to ``to_onnx`` must be the one the graph produces."""
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        model = SmallSRBuilder.build(pretrained=False, upscale_factor=2, image_size=32)
        exported = model.to_onnx(save=False)
        # The legacy TorchScript exporter (the default before torch 2.9) declares the dimensions after a resize of the
        # dynamic batch as symbolic; shape inference with data propagation reads the size either exporter's graph makes.
        inferred = onnx.shape_inference.infer_shapes(exported, data_prop=True)
        out_dim = inferred.graph.output[0].type.tensor_type.shape.dim[-1]
        assert out_dim.dim_value == model.output_image_size


class _PerImageOutputs(nn.Module):
    """A post-processor that returns a list with one ``(1, C, H, W)`` image per input image (#6161)."""

    def forward(self, imgs):
        return [imgs[i : i + 1] for i in range(imgs.shape[0])]


class _ImagesOneByOne(nn.Module):
    """A pre-processor that returns a list of ``(C, H, W)`` images, which ``forward`` runs one at a time (#6161)."""

    def forward(self, imgs):
        return [imgs[i] for i in range(imgs.shape[0])]


class _Doubled(nn.Module):
    """A post-processor of the caller's own that returns one tensor for the batch."""

    def forward(self, imgs):
        return 2 * imgs


class _DrawingNormalisedModel(nn.Module):
    """A convolution behind a batch norm, which a forward pass in training mode would update, drawing in any mode."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.norm = nn.BatchNorm2d(3)

    def forward(self, x):
        return self.norm(self.conv(x + 0.0 * torch.rand_like(x)))


def _small_super_resolution(post_processor: nn.Module, pre_processor: Optional[nn.Module] = None) -> SuperResolution:
    """A one-convolution model behind the RRDB builder's identity pre-processor, at a fixed 16 x 16 input."""
    torch.manual_seed(0)
    model = SuperResolution(nn.Conv2d(3, 3, 3, padding=1), pre_processor or nn.Identity(), post_processor)
    model.input_image_size = model.output_image_size = 16
    return model


def _count_probes(monkeypatch):
    """Count the runs that read the model's output kind, wherever ``to_onnx`` or its check makes them."""
    mixin = importlib.import_module("kornia.core.mixin.onnx")
    probe = mixin._returns_list
    calls = []

    def _counting(module, inputs):
        calls.append(type(module).__name__)
        return probe(module, inputs)

    monkeypatch.setattr(kornia.contrib.super_resolution, "_returns_list", _counting)
    monkeypatch.setattr(mixin, "_returns_list", _counting)
    return calls


def _run_onnx(op, images):
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"input": images.numpy()})


def _importorskip_exporter(export_kwargs):
    pytest.importorskip("onnx")
    if export_kwargs.get("dynamo", torch_version_ge(2, 9, 0)):
        pytest.importorskip("onnxscript")


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


@pytest.mark.device_agnostic
class TestSuperResolutionONNXBatch(BaseTester):
    """A post-processor that returns one image per input image fixes the exported batch (#6161)."""

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EACH_EXPORTER)
    def test_per_image_network_exports_one_image_6197(self, export_kwargs):
        _importorskip_exporter(export_kwargs)
        ort = pytest.importorskip("onnxruntime")
        model = SuperResolution(_PerImageOutputs(), nn.Identity(), nn.Identity())
        model.input_image_size = model.output_image_size = 16
        op = model.to_onnx(save=False, **export_kwargs)
        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
        images = torch.rand(3, 3, 16, 16)
        outputs = _run_onnx(op, images[:1])
        assert len(outputs) == 1
        self.assert_close(torch.from_numpy(outputs[0]), images[:1])
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
            _run_onnx(op, images)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("override", _DYNAMIC_BATCH)
    def test_per_image_network_rejects_dynamic_export_6197(self, override, tmp_path):
        _importorskip_exporter(override)
        model = SuperResolution(_PerImageOutputs(), nn.Identity(), nn.Identity())
        model.input_image_size = model.output_image_size = 16
        path = tmp_path / "sr.onnx"
        with pytest.raises(ValueError, match="returns one output per image"):
            model.to_onnx(onnx_name=str(path), **override)
        assert not path.exists()

    @pytest.mark.timeout(120)
    def test_batch_dependent_output_rejects_dynamic_export_6197(self, tmp_path):
        pytest.importorskip("onnx")

        class BatchDependentOutputs(nn.Module):
            def forward(self, images):
                return images if images.shape[0] == 1 else list(images.split(1))

        model = _small_super_resolution(BatchDependentOutputs())
        path = tmp_path / "sr.onnx"
        with pytest.raises(ValueError, match="returns one output per image"):
            model.to_onnx(onnx_name=str(path), dynamo=False)
        assert not path.exists()

    @pytest.mark.parametrize("component", ["model", "pre_processor", "post_processor", "wrapper"])
    @pytest.mark.parametrize("override", ["subclass", "instance"])
    def test_forward_overrides_disable_tensor_shortcut_6197(self, component, override):
        class ListIdentity(nn.Identity):
            def forward(self, images):
                return list(images.split(1))

        class ListSuperResolution(SuperResolution):
            def forward(self, images):
                return list(images.split(1))

        model = SuperResolution(nn.Identity(), nn.Identity(), nn.Identity())
        if component == "wrapper":
            if override == "subclass":
                model = ListSuperResolution(nn.Identity(), nn.Identity(), nn.Identity())
            else:
                model.forward = ListIdentity().forward
        elif override == "subclass":
            setattr(model, component, ListIdentity())
        else:
            getattr(model, component).forward = ListIdentity().forward
        assert not model._returns_one_tensor_without_a_probe()

    @pytest.mark.parametrize("component", ["model", "pre_processor", "post_processor", "wrapper"])
    def test_forward_hooks_disable_tensor_shortcut_6197(self, component):
        model = SuperResolution(nn.Identity(), nn.Identity(), nn.Identity())
        target = model if component == "wrapper" else getattr(model, component)
        target.register_forward_hook(lambda module, inputs, output: list(output.split(1)))
        assert not model._returns_one_tensor_without_a_probe()

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EACH_EXPORTER)
    def test_per_image_post_processor_exports_one_image_6161(self, export_kwargs):
        # The exporters turn a list into one graph output per traced image. Exported for a dynamic batch, the
        # TorchScript graph returned the traced batch's two images at any batch size, and the dynamo exporter failed;
        # such a model is exported for one image, a batch that ONNX Runtime enforces.
        _importorskip_exporter(export_kwargs)
        ort = pytest.importorskip("onnxruntime")
        model = _small_super_resolution(_PerImageOutputs())
        images = torch.rand(3, 3, 16, 16, generator=torch.Generator().manual_seed(0))

        op = model.to_onnx(save=False, **export_kwargs)

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
        assert op.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 1
        outputs = _run_onnx(op, images[:1])
        assert len(outputs) == 1
        with torch.no_grad():
            expected = model(images[:1])
        self.assert_close(torch.from_numpy(outputs[0]), expected[0], rtol=0.0, atol=1e-5)
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
            _run_onnx(op, images)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("override", _DYNAMIC_BATCH)
    def test_per_image_post_processor_rejects_a_dynamic_batch_6161(self, override, tmp_path):
        # A caller's dynamic batch would give back the traced batch's images at any batch size: the export raises
        # after it runs, before anything is saved, and names the cause.
        _importorskip_exporter(override)
        model = _small_super_resolution(_PerImageOutputs())
        path = tmp_path / "sr.onnx"

        with pytest.raises(ValueError, match="SuperResolution returns one output per image") as info:
            model.to_onnx(onnx_name=str(path), **override)

        assert "OutputRangePostProcessor" in str(info.value)
        assert not path.exists()

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", _EACH_EXPORTER)
    @pytest.mark.parametrize(
        "post_processor", [nn.Identity, OutputRangePostProcessor], ids=["identity", "output_range"]
    )
    def test_per_image_pre_processor_exports_one_image_6161(self, post_processor, export_kwargs):
        # A pre-processor that returns a list of images makes ``forward`` run the network one image at a time, and
        # the builders' post-processors pass the list on: such a model is exported for one image too.
        _importorskip_exporter(export_kwargs)
        if export_kwargs["dynamo"] and torch_version_lt(2, 9, 0):
            # Measured on torch 2.5.1 (2.6 to 2.8 not checked), on main as well: after other exports in the process, an
            # eager call of this model followed by its dynamo export can fail with "Hit PythonDispatcher dispatch key
            # but PythonDispatcherTLS was not set", and later dynamo exports in the process with it. Which
            # parametrization fails depends on what ran before, so both are skipped.
            pytest.skip("torch < 2.9: the dynamo exporter's PythonDispatcher state can break on this graph")
        model = _small_super_resolution(post_processor(), _ImagesOneByOne())
        images = torch.rand(3, 3, 16, 16, generator=torch.Generator().manual_seed(0))

        op = model.to_onnx(save=False, **export_kwargs)

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
        outputs = _run_onnx(op, images[:1])
        assert len(outputs) == 1
        with torch.no_grad():
            expected = model(images[:1])
        self.assert_close(torch.from_numpy(outputs[0]), expected[0], rtol=0.0, atol=1e-5)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("override", _DYNAMIC_BATCH)
    @pytest.mark.parametrize(
        "post_processor", [nn.Identity, OutputRangePostProcessor], ids=["identity", "output_range"]
    )
    def test_per_image_pre_processor_rejects_a_dynamic_batch_6161(self, post_processor, override):
        _importorskip_exporter(override)
        model = _small_super_resolution(post_processor(), _ImagesOneByOne())

        with pytest.raises(ValueError, match="SuperResolution returns one output per image"):
            model.to_onnx(save=False, **override)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", [*_EACH_EXPORTER, *_DYNAMIC_BATCH])
    @pytest.mark.parametrize(
        "post_processor", [nn.Identity, OutputRangePostProcessor], ids=["identity", "output_range"]
    )
    @pytest.mark.parametrize(
        "pre_processor", [nn.Identity, lambda: ResizePreProcessor(16, 16)], ids=["identity", "resize"]
    )
    def test_builders_processors_keep_a_dynamic_batch_without_a_probe_6161(
        self, pre_processor, post_processor, export_kwargs, monkeypatch
    ):
        # The builders' pre-processors hand the network one tensor and their post-processors keep it one: no forward
        # pass reads the output kind, before the export or after it, which for an RRDB model would cost more than the
        # export.
        _importorskip_exporter(export_kwargs)

        def _no_probe(module, inputs):
            raise AssertionError("to_onnx ran the model to read its output kind")

        monkeypatch.setattr(kornia.contrib.super_resolution, "_returns_list", _no_probe)
        monkeypatch.setattr(importlib.import_module("kornia.core.mixin.onnx"), "_returns_list", _no_probe)
        model = _small_super_resolution(post_processor(), pre_processor())

        op = model.to_onnx(save=False, **export_kwargs)

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        images = torch.rand(3, 3, 16, 16, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            expected = model(images)
        self.assert_close(torch.from_numpy(_run_onnx(op, images)[0]), expected, rtol=0.0, atol=1e-5)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("export_kwargs", [*_EACH_EXPORTER, *_DYNAMIC_BATCH])
    def test_tensor_post_processor_keeps_a_dynamic_batch_6161(self, export_kwargs):
        # A post-processor of the caller's own that returns one tensor for the batch is read as such: the batch stays
        # dynamic, also through a caller's dynamic batch.
        _importorskip_exporter(export_kwargs)
        model = _small_super_resolution(_Doubled())

        op = model.to_onnx(save=False, **export_kwargs)

        assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        images = torch.rand(3, 3, 16, 16, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            expected = model(images)
        self.assert_close(torch.from_numpy(_run_onnx(op, images)[0]), expected, rtol=0.0, atol=1e-5)

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        ("post_processor", "override"),
        [
            pytest.param(_Doubled, {"dynamo": False}, id="tensor_torchscript"),
            pytest.param(_PerImageOutputs, {"dynamo": False}, id="list_torchscript"),
            pytest.param(
                _PerImageOutputs,
                {"dynamo": False, "dynamic_axes": {"input": {0: "batch"}}},
                id="list_torchscript_batch_override",
            ),
        ],
    )
    def test_rechecks_tensor_output_at_the_tracing_batch_6161(self, post_processor, override, monkeypatch):
        # A list result already requires a fixed batch. A tensor result is checked again on the tracing batch.
        # Neither reading is kept after the export.
        _importorskip_exporter(override)
        calls = _count_probes(monkeypatch)
        model = _small_super_resolution(post_processor())

        if "dynamic_axes" in override:
            with pytest.raises(ValueError, match="returns one output per image"):
                model.to_onnx(save=False, **override)
        else:
            model.to_onnx(save=False, **override)

        assert calls == ["SuperResolution"] * (2 if post_processor is _Doubled else 1)
        assert "_onnx_list_output" not in model.__dict__

    @pytest.mark.timeout(120)
    def test_output_probe_leaves_the_model_as_it_was_6161(self):
        # The run that reads the output kind is in eval mode, on the network's dtype, without gradients: the batch
        # norm keeps its statistics, each submodule its training flag, and the global random state is given back.
        pytest.importorskip("onnx")

        def _draws_after(export):
            torch.manual_seed(0)
            network = _DrawingNormalisedModel()
            model = SuperResolution(network, nn.Identity(), _PerImageOutputs()).double().train()
            model.input_image_size = model.output_image_size = 16
            model.pre_processor.eval()  # a mixed state, which ``train()`` would not restore
            running_mean = network.norm.running_mean.clone()
            if export:
                op = model.to_onnx(save=False, dynamo=False)
                assert op.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
                assert torch.equal(network.norm.running_mean, running_mean)
                assert network.norm.num_batches_tracked.item() == 0
                assert model.training
                assert network.training
                assert network.norm.training
                assert not model.pre_processor.training
            return torch.rand(8)

        self.assert_close(_draws_after(True), _draws_after(False))


class TestRRDBNetBuilderWeights:
    """Pretrained Real-ESRGAN weights load like every other kornia checkpoint, with no download here."""

    @staticmethod
    def _fake_loader(monkeypatch, checkpoint):
        import kornia.contrib.super_resolution as sr

        calls = []

        def _load(url, **kwargs):
            calls.append((url, kwargs))
            return checkpoint

        monkeypatch.setattr(sr, "load_state_dict_from_url", _load)
        return calls

    @pytest.mark.parametrize("keys", [("params_ema",), ("params",), ("params", "params_ema")])
    def test_loads_the_ema_weights_and_falls_back_to_params(self, monkeypatch, tmp_path, keys):
        # Real-ESRGAN's RealESRGANer prefers ``params_ema`` and falls back to ``params``; kornia read ``params_ema``
        # only, so a checkpoint without it raised ``KeyError``.
        monkeypatch.chdir(tmp_path)
        reference = RRDBNetBuilder.build("RealESRGAN_x4plus_anime_6B", pretrained=False).model.state_dict()
        checkpoint = {key: {k: torch.full_like(v, i + 1.0) for k, v in reference.items()} for i, key in enumerate(keys)}
        calls = self._fake_loader(monkeypatch, checkpoint)

        model = RRDBNetBuilder.build("RealESRGAN_x4plus_anime_6B", pretrained=True).model

        expected = checkpoint["params_ema" if "params_ema" in keys else "params"]
        assert all(torch.equal(v, expected[k]) for k, v in model.state_dict().items())
        assert [url for url, _ in calls] == [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]
        # The weights go to the torch hub cache, as every other kornia checkpoint does, not under the working
        # directory, where each new one downloaded them again: no ``model_dir`` overrides the hub's.
        assert calls[0][1].get("model_dir") is None
        assert calls[0][1].get("file_name") is None
        assert not any(tmp_path.iterdir())
