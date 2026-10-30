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

"""Export and runtime contract of the ONNX mixins, checked end to end through onnxruntime."""

import ast
import copy
import pickle
from pathlib import Path

import pytest
import torch
from torch import nn

import kornia
from kornia.core import ImageSequential
from kornia.core._compat import torch_version_ge
from kornia.core.mixin.onnx import ONNXMixin, ONNXRuntimeMixin
from kornia.filters import GaussianBlur2d

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
if torch_version_ge(2, 9, 0):
    # From torch 2.9 ``torch.onnx.export`` defaults to the dynamo exporter, which needs onnxscript.
    pytest.importorskip("onnxscript")

# The opset ``to_onnx`` documents and requests.
DOCUMENTED_OPSET = 18

_RGB = {"input_shape": [-1, 3, -1, -1], "output_shape": [-1, 3, -1, -1]}


def _default_domain_opset(op):
    return next(entry.version for entry in op.opset_import if entry.domain in ("", "ai.onnx"))


def _input_dims(op):
    return [d.dim_param or d.dim_value for d in op.graph.input[0].type.tensor_type.shape.dim]


_NO_CACHE = object()


def _cache_of(module):
    # The ``show()`` / ``save()`` output cache as stored on the module, or a sentinel when it has none.
    return module.__dict__.get("_output_image", _NO_CACHE)


# What a module keeps from its last call and reads back afterwards (#6140): the output cache of ``show()`` / ``save()``;
# the parameters ``inverse()`` replays; the transformation matrix, built or lazy, and the params dict and values it was
# computed from, which the containers' annotation handlers match against; a container's per-child matrices; and
# ``AugmentationSequential``'s input and mask dtypes.
_CALL_STATE = (
    "_output_image",
    "_params",
    "_transform_matrix",
    "_lazy_matrix_args",
    "_transform_matrix_params",
    "_transform_matrix_params_snapshot",
    "_transform_matrices",
    "input_dtype",
    "mask_dtype",
)


def _call_state_of(module):
    # Each of those attributes as stored on the module itself; a class-level default is not an entry.
    return {name: module.__dict__[name] for name in _CALL_STATE if name in module.__dict__}


def _same_objects(state, before):
    return state.keys() == before.keys() and all(state[name] is before[name] for name in state)


def _composed(matrices):
    # The product a container's lazy ``transform_matrix`` builds from its per-child matrices, in execution order.
    product = matrices[0]
    for matrix in matrices[1:]:
        product = matrix @ product
    return product


# What an augmentation or image module writes on itself outside ``__init__`` and ``make_samplers`` that is not state of
# its last call, so ``to_onnx`` leaves it alone: per writing method, the attributes and the reason. A name in
# ``_PER_CALL_STATE`` is exempt in every method, since ``to_onnx`` restores it on any module; any other name is
# classified for the method that writes it, so the same name written by a new method (``self.p`` in a ``forward``)
# fails the test. Writes the scan does not see: ``setattr`` and ``self.__dict__[...]``, a write through another name
# (a container setting ``mod.keepdim``), and a ``for self.x in`` target.
_NOT_CALL_STATE = {
    "ColorJitter.compile": (
        ("_brightness_fn", "_contrast_fn", "_saturation_fn", "_hue_fn", "_cond_fn"),
        "the compiled kernels",
    ),
    "RandomGaussianBlur.compile": (("_gaussian_blur2d_fn",), "the compiled kernel"),
    "RandomGaussianIllumination.compile": (("_fn",), "the compiled kernel"),
    "_PicklableCompileMixin._record_compile": (("_compile_kwargs", "_uncompiled_fns"), "compile() records its options"),
    "_PicklableCompileMixin.__setstate__": (("_uncompiled_fns",), "unpickling"),
    "AugmentationSequentialOps.data_keys": (
        ("_data_keys",),
        "a property setter, which the container calls to set its data keys back before each call returns",
    ),
    "ImageModuleForSequentialMixIn.disable_features": (("_disable_features",), "a property setter"),
    "ImageModule.disable_features": (("_disable_features",), "a property setter"),
    "ImageSequential.disable_features": (("_disable_features",), "a property setter"),
    "SequentialBase.keepdim": (("_keepdim",), "a property setter"),
    "SequentialBase.same_on_batch": (("_same_on_batch",), "a property setter"),
    "OperationBase._init_magnitude": (("_factor_name", "_magnitude", "magnitude_range"), "a constructor helper"),
    "TransformMatrixMinIn._parse_transformation_matrix_mode": (("_transformation_matrix_arg",), "a constructor helper"),
    "ImageSequentialBase.transform_inputs": (
        ("_sequence_matrices",),
        "the enclosing AugmentationSequential reads it right after this image call, in the same call",
    ),
    "_BasicAugmentationBase.set_rng_device_and_dtype": (("device", "dtype"), "called by .to(), not by a call"),
    "RandomGeneratorBase.set_rng_device_and_dtype": (("device", "dtype"), "called by .to(), not by a call"),
    "RandomTransplantation.params_from_input": (
        ("excluded_labels",),
        "a constructor argument moved to the mask's device; its values never change",
    ),
    "AdaptiveDiscriminatorAugmentation.update": (
        ("p", "_num_calls", "real_acc_ema"),
        "updated only from an explicit real_acc, which the export does not pass",
    ),
}

_SCANNED = ("augmentation", "core/module.py", "core/mixin/image_module.py")


def _attributes_written_after_construction():
    # ``self.<name> = ...`` (also augmented and annotated assignments) in every method of every class in the scanned
    # sources but ``__init__`` and the generators' ``make_samplers``, as ``(Class.method, name, path:line)``.
    root = Path(kornia.__file__).parent
    writes = []
    for scanned in _SCANNED:
        paths = sorted((root / scanned).rglob("*.py")) if (root / scanned).is_dir() else [root / scanned]
        for path in paths:
            for cls in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(cls, ast.ClassDef):
                    continue
                for method in cls.body:
                    if not isinstance(method, ast.FunctionDef) or method.name in ("__init__", "make_samplers"):
                        continue
                    if not method.args.args or method.args.args[0].arg != "self":
                        continue  # a static method, a class method or ``__init_subclass__``
                    for node in ast.walk(method):
                        if isinstance(node, ast.Assign):
                            targets = node.targets
                        elif isinstance(node, ast.AugAssign) or (isinstance(node, ast.AnnAssign) and node.value):
                            targets = [node.target]
                        else:
                            continue
                        for target in targets:
                            for item in target.elts if isinstance(target, ast.Tuple) else [target]:
                                if (
                                    isinstance(item, ast.Attribute)
                                    and isinstance(item.value, ast.Name)
                                    and item.value.id == "self"
                                ):
                                    where = f"{path.relative_to(root.parent).as_posix()}:{node.lineno}"
                                    writes.append((f"{cls.name}.{method.name}", item.attr, where))
    return writes


def _run(op, x):
    session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])
    return torch.from_numpy(session.run(None, {session.get_inputs()[0].name: x.numpy()})[0])


@pytest.mark.device_agnostic
class TestONNXExportMixin(BaseTester):
    def test_train_mode_model_exports_the_eval_graph(self):
        torch.manual_seed(0)
        model = ImageSequential(nn.Conv2d(3, 3, 3, padding=1), nn.BatchNorm2d(3), nn.Dropout(0.5))
        model.train()
        model[1].eval()  # a mixed state, which must come back exactly
        op = model.to_onnx(save=False, **_RGB)

        assert "Dropout" not in {node.op_type for node in op.graph.node}
        assert [m.training for m in (model, *model)] == [True, True, False, True]

        x = torch.rand(2, 3, 8, 8)
        with torch.no_grad():
            expected = copy.deepcopy(model).eval()(x)
        self.assert_close(_run(op, x), expected)

    def test_failed_export_restores_the_training_mode(self, monkeypatch):
        model = ImageSequential(nn.Conv2d(3, 3, 1), nn.Dropout(0.5))
        model.train()
        model[0].eval()  # a mixed state: ``model.train(True)`` in place of an exact restore would lose it

        def _fail(*args, **kwargs):
            raise RuntimeError("export failed")

        monkeypatch.setattr(torch.onnx, "export", _fail)
        with pytest.raises(RuntimeError, match="export failed"):
            model.to_onnx(save=False, **_RGB)
        assert [m.training for m in (model, *model)] == [True, False, True]

    @staticmethod
    def _cached_tree():
        # Every image module caches its last output for ``show()`` / ``save()``: a nested ``ImageModule`` through
        # ``_store_output_image``, an ``AugmentationSequential`` by writing the attribute itself.
        import kornia.augmentation as K
        from kornia.core import ImageModule

        class _Child(ImageModule):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)

            def forward(self, x):
                return self.conv(x)

        child, aug = _Child(), K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0))
        model = ImageSequential(child, aug, nn.Identity())
        model(torch.rand(1, 3, 8, 8))
        fresh = ImageSequential(nn.Identity())  # never called: its cache is None
        outer = ImageSequential(model, fresh)
        outer._output_image = torch.rand(1, 3, 8, 8)  # the exported module's own cache, set as a call would
        return outer, [outer, model, child, aug, fresh]

    # Not "dynamo" in the ids: conftest deselects every test whose name contains it unless an optimizer is set.
    @pytest.mark.parametrize("dynamo", [True, False], ids=["default-exporter", "legacy-exporter"])
    def test_export_keeps_every_output_cache(self, dynamo):
        # The legacy exporter runs the module on the dummy input outside ``is_exporting()``: each image module in the
        # tree used to cache the dummy's (2, 3, 4, 4) output, so ``show()`` rendered noise after an export.
        if dynamo and not torch_version_ge(2, 9, 0):
            pytest.skip("the default exporter below torch 2.9 is the legacy one")
        outer, modules = self._cached_tree()
        before = [_cache_of(m) for m in modules]
        outer.to_onnx(save=False, pseudo_shape=[1, 3, 4, 4], dynamo=dynamo, **_RGB)
        assert all(_cache_of(m) is cache for m, cache in zip(modules, before))

    def test_failed_export_keeps_every_output_cache(self, monkeypatch):
        outer, modules = self._cached_tree()
        before = [_cache_of(m) for m in modules]
        bare = nn.Identity()  # a module without the attribute must not get one
        outer.append(bare)

        def _run_then_fail(module, args, *rest, **kwargs):
            module(*args) if isinstance(args, tuple) else module(args)
            bare._output_image = "written by the export"
            raise RuntimeError("export failed")

        monkeypatch.setattr(torch.onnx, "export", _run_then_fail)
        with pytest.raises(RuntimeError, match="export failed"):
            outer.to_onnx(save=False, pseudo_shape=[1, 3, 4, 4], **_RGB)
        assert all(_cache_of(m) is cache for m, cache in zip(modules, before))
        assert _cache_of(bare) is _NO_CACHE

    @staticmethod
    def _augmented(case):
        # The pipeline of #6140: flips drawn per sample, called on one 8 x 8 image. The export traces a batch of two
        # 4 x 4 dummies, whose draws then replaced the state ``inverse()`` reads, so it raised or inverted the dummy.
        # ``exported`` is what ``to_onnx`` exports: the ``ImageSequential`` around a container, or the bare module.
        import kornia.augmentation as K

        torch.manual_seed(3)
        if case == "container":
            aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=0.5), K.RandomVerticalFlip(p=0.5))
            exported = ImageSequential(aug)
        else:
            aug = exported = K.RandomHorizontalFlip(p=0.5)
        y = exported(torch.rand(1, 3, 8, 8))
        return exported, aug, y, aug.inverse(y)

    @staticmethod
    def _export(exported, **kwargs):
        # A bare augmentation module has no ``to_onnx``; it is exported as the ``model`` of an empty receiver.
        receiver = exported if isinstance(exported, ImageSequential) else ImageSequential()
        model = None if receiver is exported else exported
        return receiver.to_onnx(save=False, model=model, pseudo_shape=[1, 3, 4, 4], **_RGB, **kwargs)

    # Not "dynamo" in the ids: conftest deselects every test whose name contains it unless an optimizer is set. The
    # default exporter is torch.export-based from torch 2.9 and the legacy one before.
    @pytest.mark.parametrize("case", ["container", "module"], ids=["nested-container", "bare-module"])
    @pytest.mark.parametrize("exporter", [{}, {"dynamo": False}], ids=["default-exporter", "legacy-exporter"])
    def test_export_keeps_the_state_inverse_reads_6140(self, exporter, case):
        exported, aug, y, expected = self._augmented(case)
        modules = list(exported.modules())
        params = aug._params
        before = [_call_state_of(m) for m in modules]

        self._export(exported, **exporter)

        assert aug._params is params
        assert all(_same_objects(_call_state_of(m), state) for m, state in zip(modules, before))
        self.assert_close(aug.inverse(y), expected)

    @pytest.mark.parametrize("exporter", [{}, {"dynamo": False}], ids=["default-exporter", "legacy-exporter"])
    def test_export_leaves_no_state_on_a_module_never_called_6140(self, exporter):
        # Several of the attributes are class-level defaults until a call stores them on the instance.
        import kornia.augmentation as K

        exported = ImageSequential(K.AugmentationSequential(K.RandomHorizontalFlip(p=0.5)))
        modules = list(exported.modules())
        before = [_call_state_of(m) for m in modules]

        self._export(exported, **exporter)

        assert all(_same_objects(_call_state_of(m), state) for m, state in zip(modules, before))

    def test_failed_export_keeps_the_state_inverse_reads_6140(self, monkeypatch):
        exported, aug, y, expected = self._augmented("container")
        modules = list(exported.modules())
        before = [_call_state_of(m) for m in modules]
        bare = nn.Identity()  # a module without the state must not get it
        exported.append(bare)

        def _run_then_fail(module, args, *rest, **kwargs):
            module(*args) if isinstance(args, tuple) else module(args)
            bare._params = "written by the export"
            raise RuntimeError("export failed")

        monkeypatch.setattr(torch.onnx, "export", _run_then_fail)
        with pytest.raises(RuntimeError, match="export failed"):
            self._export(exported)
        assert all(_same_objects(_call_state_of(m), state) for m, state in zip(modules, before))
        assert _call_state_of(bare) == {}
        self.assert_close(aug.inverse(y), expected)

    # A static shape: with a dynamic batch, the dynamo exporter stops in the generators before the modules below run.
    _STATIC = {"input_shape": [2, 3, 4, 4], "pseudo_shape": [2, 3, 4, 4]}

    @pytest.mark.parametrize("exporter", [{}, {"dynamo": False}], ids=["default-exporter", "legacy-exporter"])
    def test_export_keeps_the_matrix_of_a_policy_6140(self, exporter):
        # Under a capture, where ``is_exporting()`` skips the reset, a policy appends the dummy's matrices in place to
        # the list its ``transform_matrix`` composes, so putting the same list back is not enough.
        from kornia.augmentation.auto.operations import Rotate, ShearX
        from kornia.augmentation.auto.operations.policy import PolicySequential

        torch.manual_seed(0)
        policy = PolicySequential(ShearX(initial_probability=1.0), Rotate(initial_probability=1.0))
        policy(torch.rand(1, 3, 8, 8))
        matrices = policy._transform_matrices
        items = list(matrices)
        expected = _composed(items)  # read lazily, so the property must not run before the export

        # The legacy exporter (the default before torch 2.9) lowers the warp's ``affine_grid`` from opset 20 only.
        legacy = exporter.get("dynamo") is False or not torch_version_ge(2, 9, 0)
        opset = {"opset_version": 20} if legacy else {}
        ImageSequential().to_onnx(
            save=False, model=policy, output_shape=[2, 3, 4, 4], **self._STATIC, **exporter, **opset
        )

        assert policy._transform_matrices is matrices
        assert len(matrices) == len(items)
        assert all(now is item for now, item in zip(matrices, items))
        self.assert_close(policy.transform_matrix, expected)

    def test_failed_export_keeps_the_matrices_a_policy_appends_in_place_6140(self, monkeypatch):
        # An auto-augment policy and its sub-policy append in place too. No real exporter reaches them today (their
        # draws stop torch.export first), so the stand-in export runs the model with ``is_exporting()`` True, as a
        # capture does, then fails.
        import kornia.augmentation as K
        import kornia.core.utils

        torch.manual_seed(0)
        policy = K.auto.AutoAugment(policy=[[("shear_x", 1.0, 5), ("rotate", 1.0, 5)]])
        policy(torch.rand(1, 3, 8, 8))
        lists = [
            (m, m._transform_matrices, list(m._transform_matrices))
            for m in policy.modules()
            if m.__dict__.get("_transform_matrices")
        ]
        expected = _composed(policy._transform_matrices)

        def _capture_then_fail(module, args, *rest, **kwargs):
            with monkeypatch.context() as patch:
                patch.setattr(kornia.core.utils, "_is_exporting_eager", lambda: True)
                module(*args) if isinstance(args, tuple) else module(args)
            raise RuntimeError("export failed")

        monkeypatch.setattr(torch.onnx, "export", _capture_then_fail)
        with pytest.raises(RuntimeError, match="export failed"):
            ImageSequential().to_onnx(save=False, model=policy, output_shape=[2, 3, 4, 4], **self._STATIC)

        assert len(lists) == 2  # the policy and its one sub-policy
        for module, matrices, items in lists:
            assert module._transform_matrices is matrices
            assert len(matrices) == len(items)
            assert all(now is item for now, item in zip(matrices, items))
        self.assert_close(policy.transform_matrix, expected)

    def test_failed_export_gives_back_the_items_of_a_params_dict_6140(self, monkeypatch):
        # ``_params`` and ``_transform_matrix_params`` are dicts that ``forward`` edits in place when they are passed
        # back to it (``params["batch_prob"] = ...``, ``save_kwargs=True``). No export run does that today; the stand-in
        # export edits the stored dict in place, then fails.
        exported, aug, y, expected = self._augmented("module")
        params = aug._params
        items = dict(params)

        def _edit_then_fail(module, args, *rest, **kwargs):
            module._params["batch_prob"] = torch.zeros(2)
            module._params["written by the export"] = True
            raise RuntimeError("export failed")

        monkeypatch.setattr(torch.onnx, "export", _edit_then_fail)
        with pytest.raises(RuntimeError, match="export failed"):
            self._export(exported)

        assert aug._params is params
        assert params.keys() == items.keys()
        assert all(params[key] is items[key] for key in items)
        self.assert_close(aug.inverse(y), expected)

    @pytest.mark.parametrize("exporter", [{}, {"dynamo": False}], ids=["default-exporter", "legacy-exporter"])
    def test_export_leaves_a_resized_crop_copyable_6140(self, exporter):
        # ``ResizedCropGenerator`` stored each call's crop sizes on itself. A dynamo export on torch 2.9 left its fake
        # tensor there, and ``deepcopy`` and ``pickle`` raised afterwards. The sizes are now passed on, not stored.
        import kornia.augmentation as K

        torch.manual_seed(0)
        aug = K.RandomResizedCrop((6, 6), p=1.0)
        aug(torch.rand(1, 3, 8, 8))

        ImageSequential().to_onnx(save=False, model=aug, output_shape=[2, 3, 6, 6], **self._STATIC, **exporter)

        assert aug._param_generator.size == (6, 6)
        copy.deepcopy(aug)
        pickle.dumps(aug)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_failed_export_leaves_no_buffer_registration_hook_6140(self):
        # torch 2.9's non-strict torch.export registers a global buffer-registration hook and removes it only when its
        # trace succeeds. Left behind by a failed export, it ran in every later Dynamo trace that registers a buffer,
        # which then raised ``Unsupported``: on CI, the losses' traces after this file.
        class _Fails(nn.Module):
            def forward(self, x):
                raise ValueError("this model does not export")

        class _WithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.ones(1))

        hooks = getattr(torch.nn.modules.module, "_global_buffer_registration_hooks", {})
        before = list(hooks)
        with pytest.raises(Exception) as info:  # each exporter wraps the model's error in its own
            ImageSequential(_Fails()).to_onnx(save=False, **_RGB)
        error = info.value
        while error is not None and not isinstance(error, ValueError):
            error = error.__cause__ or error.__context__
        assert error is not None, f"no ValueError behind {info.value!r}"
        assert "does not export" in str(error)
        assert list(hooks) == before

        torch._dynamo.reset()
        traced = torch.compile(lambda x: x + _WithBuffer().offset, backend="eager", fullgraph=True)
        self.assert_close(traced(torch.zeros(2)), torch.ones(2))

    def test_every_attribute_written_after_construction_is_classified_6140(self):
        # ``to_onnx`` restores the attributes in ``_PER_CALL_STATE``. An augmentation or image module that starts
        # writing another attribute on itself outside its constructor must say which kind it is: state of its last
        # call, which the export's run on the dummy would overwrite, or one of the kinds in ``_NOT_CALL_STATE``.
        from kornia.core.mixin.onnx import _PER_CALL_STATE

        writes = _attributes_written_after_construction()
        unclassified = sorted(
            (writer, name, where)
            for writer, name, where in writes
            if name not in _PER_CALL_STATE and name not in _NOT_CALL_STATE.get(writer, ((), ""))[0]
        )
        assert not unclassified, (
            "Add each to _PER_CALL_STATE in kornia/core/mixin/onnx.py if a later call or inverse() reads what the last "
            "call stored, else to _NOT_CALL_STATE here under its writer with the reason: "
            + "; ".join(f"{writer} writes {name} ({where})" for writer, name, where in unclassified)
        )
        assert set(_PER_CALL_STATE) == set(_CALL_STATE)
        written = {(writer, name) for writer, name, _ in writes}
        classified = {(writer, name) for writer, (names, _) in _NOT_CALL_STATE.items() for name in names}
        stale = sorted(classified - written) + sorted(set(_PER_CALL_STATE) - {name for _, name in written})
        assert not stale, f"no longer written outside a constructor, remove them: {stale}"

    def test_explicit_training_mode_is_the_legacy_exporters_opt_out(self):
        # The documented opt-out from the eval-mode export: an explicit ``training=TRAINING`` reaches the legacy
        # exporter, which traces the training graph. The flags still come back exactly.
        model = ImageSequential(nn.Conv2d(3, 3, 1), nn.Dropout(0.5))
        model[0].eval()
        op = model.to_onnx(save=False, dynamo=False, training=torch.onnx.TrainingMode.TRAINING, **_RGB)

        assert "Dropout" in {node.op_type for node in op.graph.node}
        assert [m.training for m in (model, *model)] == [True, False, True]

    @pytest.mark.skipif(
        not torch_version_ge(2, 9, 0), reason="the torch.export-based exporter is the default from torch 2.9"
    )
    def test_default_exporter_from_torch_2_9_ignores_the_training_keyword(self):
        model = ImageSequential(nn.Conv2d(3, 3, 1), nn.Dropout(0.5))
        model.train()
        op = model.to_onnx(save=False, training=torch.onnx.TrainingMode.TRAINING, **_RGB)

        assert "Dropout" not in {node.op_type for node in op.graph.node}
        assert model.training

    @pytest.mark.parametrize(
        "channels, module",
        [(3, lambda: nn.Conv2d(3, 3, 1)), (1, lambda: GaussianBlur2d((3, 3), (1.5, 1.5)))],
        ids=["conv", "gaussian_blur"],
    )
    def test_saved_model_has_the_documented_opset(self, tmp_path, channels, module):
        """Test that the saved ONNX model uses the documented opset."""
        # The dynamo exporter builds opset 18 and down-converts a lower request only when every op has an adapter, which
        # a convolution has and the blur's ``Pad`` has not.
        path = tmp_path / "model.onnx"
        shape = [2, channels, 16, 16]
        op = ImageSequential(module()).to_onnx(onnx_name=str(path), input_shape=shape, output_shape=shape)
        assert _default_domain_opset(op) == DOCUMENTED_OPSET
        assert _default_domain_opset(onnx.load(str(path))) == DOCUMENTED_OPSET

    def test_default_input_shape_exports_a_fixed_channel_model(self):
        model = ImageSequential(nn.Conv2d(3, 3, 1))
        op = model.to_onnx(save=False)

        dims = _input_dims(op)
        assert dims[1] == 3
        assert all(isinstance(d, str) for d in (dims[0], dims[2], dims[3]))

        # the batch is traced at 2: batches 1 and 3 check that the graph does not keep it
        for batch in (1, 2, 3):
            x = torch.rand(batch, 3, 5, 7)
            with torch.no_grad():
                expected = model(x)
            self.assert_close(_run(op, x), expected)

    def test_pseudo_shape_alone_sets_the_channel_count(self):
        # Without input_shape, the default's fixed channel count follows an explicit pseudo shape.
        model = ImageSequential(nn.Conv2d(1, 1, 1))
        op = model.to_onnx(save=False, pseudo_shape=[1, 1, 32, 32])

        for batch in (1, 2, 3):
            x = torch.rand(batch, 1, 5, 7)
            with torch.no_grad():
                expected = model(x)
            self.assert_close(_run(op, x), expected)

        dims = _input_dims(op)
        assert dims[1] == 1
        assert all(isinstance(d, str) for d in (dims[0], dims[2], dims[3]))

    def test_input_shape_must_agree_with_an_explicit_pseudo_shape(self):
        with pytest.raises(ValueError, match="input_shape"):
            ImageSequential(nn.Sigmoid()).to_onnx(save=False, input_shape=[-1, 3, -1, -1], pseudo_shape=[1, 1, 32, 32])

    def test_input_shape_rank_must_match_the_pseudo_shape(self):
        with pytest.raises(ValueError, match="pseudo shape"):
            ImageSequential(nn.Identity()).to_onnx(save=False, input_shape=[-1, 3, -1, -1, -1])

    def test_missing_output_directory_is_reported_before_exporting(self, tmp_path, monkeypatch):
        def _must_not_export(*args, **kwargs):
            raise AssertionError("the export ran before the output directory was checked")

        monkeypatch.setattr(torch.onnx, "export", _must_not_export)
        with pytest.raises(FileNotFoundError, match="no_such_dir"):
            ImageSequential(nn.Identity()).to_onnx(onnx_name=str(tmp_path / "no_such_dir" / "x.onnx"), **_RGB)

    @pytest.mark.parametrize(
        "input_shape, pseudo_shape, traced",
        [
            ([-1, 3, -1, -1], [1, 3, 1, 32], [2, 3, 2, 32]),
            ([-1, 3, 16, -1], [0, 3, 16, 5], [2, 3, 16, 5]),
            ([1, 3, 16, 16], [1, 3, 16, 16], [1, 3, 16, 16]),
        ],
        ids=["sizes_of_one", "size_of_zero", "fixed_dimensions"],
    )
    def test_dynamic_dimensions_are_traced_at_two_or_more_6066(self, monkeypatch, input_shape, pseudo_shape, traced):
        # torch.export specializes an example size of 0 or 1. On torch 2.9, DexiNed exported from a batch of 1 kept
        # that batch inside its Resize nodes and failed in ONNX Runtime at any other batch size. Fixed dimensions keep
        # theirs.
        traced_shapes = []

        def _capture(model, args, *rest, **kwargs):
            traced_shapes.append(list(args.shape))
            raise RuntimeError("captured")

        monkeypatch.setattr(torch.onnx, "export", _capture)
        with pytest.raises(RuntimeError, match="captured"):
            ImageSequential(nn.Identity()).to_onnx(save=False, input_shape=input_shape, pseudo_shape=pseudo_shape)
        assert traced_shapes == [traced]


class _ChannelAffine(nn.Module):
    """A per-channel affine map, with its weights as parameters or as buffers.

    Its ops run in float16, float32 and float64 on onnxruntime's CPU provider.
    """

    def __init__(self, as_buffers: bool = False) -> None:
        super().__init__()
        weight = torch.linspace(0.5, 1.5, 3).view(1, 3, 1, 1)
        bias = torch.linspace(-0.25, 0.25, 3).view(1, 3, 1, 1)
        if as_buffers:
            self.register_buffer("weight", weight)
            self.register_buffer("bias", bias)
        else:
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class TestONNXExportMixinDtypeDevice(BaseTester):
    @pytest.mark.parametrize("as_buffers", [False, True], ids=["parameters", "buffers"])
    def test_dummy_input_follows_the_model_dtype_and_device(self, device, dtype, as_buffers):
        model = ImageSequential(_ChannelAffine(as_buffers)).to(device, dtype)
        assert len(list(model.parameters())) == (0 if as_buffers else 2)
        op = model.to_onnx(save=False, **_RGB)

        elem_types = {
            torch.float16: onnx.TensorProto.FLOAT16,
            torch.bfloat16: onnx.TensorProto.BFLOAT16,
            torch.float32: onnx.TensorProto.FLOAT,
            torch.float64: onnx.TensorProto.DOUBLE,
        }
        assert op.graph.input[0].type.tensor_type.elem_type == elem_types[dtype]
        if dtype == torch.bfloat16:
            return  # onnxruntime's CPU provider has no bfloat16 Mul, and numpy no bfloat16 to feed it

        x = torch.linspace(0, 1, 2 * 3 * 5 * 7, device=device, dtype=dtype).view(2, 3, 5, 7)
        with torch.no_grad():
            expected = model(x)
        self.assert_close(_run(op, x.cpu()), expected.cpu())


class TestONNXExportMixinRandomState(BaseTester):
    @pytest.mark.parametrize("exporter", [{}, {"dynamo": False}], ids=["default-exporter", "legacy-exporter"])
    def test_export_gives_back_the_random_state_6140(self, device, exporter):
        # The dummy and the export's run on it draw from the global generators, an augmentation moved to a device on
        # that device's, so the draws after an export were not the ones the same calls make without it (#6140).
        import kornia.augmentation as K

        def _draws_after(export):
            torch.manual_seed(3)
            model = ImageSequential(
                K.AugmentationSequential(K.RandomHorizontalFlip(p=0.5), K.RandomVerticalFlip(p=0.5))
            ).to(device)
            x = torch.rand(4, 3, 8, 8).to(device)
            model(x)
            if export:
                model.to_onnx(save=False, pseudo_shape=[1, 3, 4, 4], **_RGB, **exporter)
            return model(x), torch.rand(8, device=device), torch.rand(8)

        for after, expected in zip(_draws_after(True), _draws_after(False)):
            self.assert_close(after, expected)


@pytest.mark.device_agnostic
class TestONNXRuntimeMixin(BaseTester):
    @pytest.fixture
    def op(self):
        return ImageSequential(nn.Sigmoid()).to_onnx(save=False, **_RGB)

    def test_session_options_are_used(self, op):
        from kornia.onnx import ONNXModule, ONNXSequential

        def _options():
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            return options

        sessions = [
            ONNXRuntimeMixin()._create_session(op, session_options=_options()),
            ONNXSequential(op, session_options=_options()).get_session(),
            ONNXModule(op, session_options=_options()).get_session(),
        ]
        for session in sessions:
            level = session.get_session_options().graph_optimization_level
            assert level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        x = torch.rand(1, 3, 4, 5)
        self.assert_close(torch.from_numpy(ONNXSequential(op, session_options=_options())(x.numpy())[0]), x.sigmoid())

    def test_runtime_rejects_wrong_input_count(self, op):
        from kornia.onnx import ONNXSequential

        runtime = ONNXSequential(op)
        x = torch.rand(1, 3, 4, 5).numpy()
        with pytest.raises(ValueError, match="Expected 1 ONNX inputs, got 0"):
            runtime()
        with pytest.raises(ValueError, match="Expected 1 ONNX inputs, got 2"):
            runtime(x, x)

    def test_runtime_feeds_inputs_in_session_order(self):
        from kornia.onnx import ONNXModule, ONNXSequential

        def _value(name):
            return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1])

        node = onnx.helper.make_node("Sub", ["x", "y"], ["output"])
        graph = onnx.helper.make_graph([node], "sub", [_value("x"), _value("y")], [_value("output")])
        op = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9)
        a, b = torch.tensor([2.0]).numpy(), torch.tensor([99.0]).numpy()
        for runtime in (ONNXSequential(op), ONNXModule(op)):
            assert runtime(a, b)[0].tolist() == [-97.0]
            assert runtime(b, a)[0].tolist() == [97.0]
            with pytest.raises(ValueError, match="Expected 2 ONNX inputs, got 3"):
                runtime(a, b, a)
            # torch tensors on the CPU and numpy arrays mix
            assert runtime(torch.tensor([2.0]), b)[0].tolist() == [-97.0]

    def test_runtime_takes_cpu_torch_tensors(self, op):
        import numpy as np

        from kornia.onnx import ONNXModule, ONNXSequential

        x = torch.rand(2, 3, 5, 4)
        inputs = [
            x,
            x.clone().requires_grad_(True),
            torch.rand(2, 3, 4, 5).transpose(-1, -2),  # not contiguous
        ]
        for runtime in (ONNXSequential(op), ONNXModule(op)):
            for value in inputs:
                (out,) = runtime(value)
                assert isinstance(out, np.ndarray)
                assert out.tolist() == runtime(np.ascontiguousarray(value.detach().numpy()))[0].tolist()
                self.assert_close(torch.from_numpy(out), value.detach().sigmoid())

    def test_runtime_rejects_tensors_off_the_cpu(self, op):
        from kornia.onnx import ONNXModule, ONNXSequential

        x = torch.empty(1, 3, 4, 5, device="meta")
        for runtime in (ONNXSequential(op), ONNXModule(op)):
            with pytest.raises(ValueError, match=r"input 0 is a torch tensor on meta.*Move it to the CPU"):
                runtime(x)

    @pytest.fixture
    def subtract(self):
        from kornia.onnx import ONNXModule

        def _value(name):
            return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [2])

        node = onnx.helper.make_node("Sub", ["x", "y"], ["output"])
        graph = onnx.helper.make_graph([node], "sub", [_value("x"), _value("y")], [_value("output")])
        return ONNXModule(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9))

    def test_runtime_names_the_input_off_the_cpu(self, subtract):
        with pytest.raises(ValueError, match=r"ONNX input 1 is a torch tensor on meta"):
            subtract(torch.tensor([3.0, 1.0]), torch.empty(2, device="meta"))

    @pytest.mark.parametrize("dtype_numpy_lacks", [torch.bfloat16, torch.float8_e4m3fn])
    def test_runtime_rejects_tensors_of_a_dtype_numpy_lacks_naming_the_input(self, subtract, dtype_numpy_lacks):
        a = torch.tensor([3.0, 1.0])
        with pytest.raises(TypeError, match=rf"ONNX input 1 is a {dtype_numpy_lacks} tensor, a dtype numpy has no"):
            subtract(a.numpy(), a.to(dtype_numpy_lacks))

    def test_runtime_rejects_sparse_tensors_naming_the_input(self, subtract):
        a = torch.tensor([3.0, 0.0])
        with pytest.raises(TypeError, match=r"ONNX input 0 is a torch.sparse_coo tensor.*\.to_dense\(\)"):
            subtract(a.to_sparse(), a)

    def test_runtime_reads_a_negative_bit_view(self, subtract):
        # .conj().imag returns a view with the negative bit set, which .numpy() refuses: [3, 1] - [-2, 1]
        negative_view = torch.tensor([1 + 2j, 3 - 1j]).conj().imag
        assert negative_view.is_neg()
        assert subtract(torch.tensor([3.0, 1.0]), negative_view)[0].tolist() == [5.0, 0.0]

    def test_add_metadata_overwrites_existing_keys(self, op):
        keys = [prop.key for prop in op.metadata_props]
        assert sorted(keys) == ["source", "version"]

        again = ONNXMixin()._add_metadata(copy.deepcopy(op), [("date", "20240909")])
        again = ONNXMixin()._add_metadata(again, [("date", "20260930")])

        assert sorted(prop.key for prop in again.metadata_props) == ["date", "source", "version"]
        assert {prop.key: prop.value for prop in again.metadata_props}["date"] == "20260930"
        onnx.checker.check_model(again)
