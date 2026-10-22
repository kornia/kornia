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

import gc
import json
import sys

import pytest
import torch
from torch import nn

import kornia
from kornia.profiler import (
    FeatureExtractor,
    ModelProfiler,
    model_profile_under_augmentation,
    model_profile_under_input_changes,
)
from kornia.profiler import model_profiler as profiler_module
from kornia.profiler.augment import AugmentationPipeline
from kornia.profiler.metrics.gram import gram_similarity
from kornia.profiler.metrics.linear import linear_cka


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Exercise the supported CPU and CUDA profiler backends when available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


@pytest.fixture
def missing_pandas(monkeypatch):
    """Simulate an unavailable optional pandas dependency without installing it."""
    monkeypatch.setitem(sys.modules, "pandas", None)
    monkeypatch.setattr(profiler_module.pandas, "module", None)
    monkeypatch.setenv("KORNIA_INSTALLATION_MODE", "raise")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cka_reference_and_invariances(device, dtype):
    """Compare CKA with a feature-space reference and its mathematical invariances."""
    torch.manual_seed(1729)
    x = torch.randn(19, 5, device=device, dtype=dtype)
    y = torch.randn(19, 7, device=device, dtype=dtype)
    # Independent feature-space formula, unlike production sample-Gram formula.
    a, b = x.double() - x.double().mean(0), y.double() - y.double().mean(0)
    reference = (a.T @ b).square().sum() / ((a.T @ a).square().sum() * (b.T @ b).square().sum()).sqrt()
    assert linear_cka(x, y) == pytest.approx(reference.item(), abs=1e-10)
    q, _ = torch.linalg.qr(torch.randn(5, 5, device=device, dtype=dtype))
    for changed in [x, x * 3, x @ q, x + torch.arange(5, device=device, dtype=dtype)]:
        assert linear_cka(x, changed) == pytest.approx(1, abs=1e-6)
    # Independent sample permutation does not generally preserve correspondence.
    assert linear_cka(x, x.flip(0)) < 0.95


@pytest.mark.parametrize("kind", ["constant", "singleton", "samples", "rank", "empty", "nan", "integer"])
def test_cka_invalid_inputs(device, kind):
    """Reject degenerate, incompatible and nonfinite CKA inputs."""
    x = torch.randn(8, 4, device=device)
    y = x.clone()
    if kind == "constant":
        y = torch.ones_like(x)
    if kind == "singleton":
        x, y = x[:1], y[:1]
    if kind == "samples":
        y = y[:3]
    if kind == "rank":
        y = y.flatten()
    if kind == "empty":
        y = y[:, :0]
    if kind == "nan":
        y[0, 0] = float("nan")
    if kind == "integer":
        y = y.to(torch.int64)
    with pytest.raises((ValueError, TypeError)):
        linear_cka(x, y)


def test_gram_all_layers_and_nonspatial(device):
    """Return every spatial layer's Gram score while ignoring vector features."""
    maps = {
        "first": torch.randn(4, 3, 8, 8, device=device),
        "second": torch.randn(4, 5, 4, 4, device=device),
        "vector": torch.randn(4, 8, device=device),
    }
    result = gram_similarity(maps, maps)
    assert set(result) == set(maps)
    assert result["first"] == pytest.approx(1, abs=1e-6)
    assert result["second"] == pytest.approx(1, abs=1e-6)
    assert result["vector"] is None
    assert gram_similarity({}, {}) == {}
    with pytest.raises(ValueError):
        gram_similarity(maps, {})


def hook_count(model):
    """Count live forward hooks across the model and its submodules."""
    return sum(len(m._forward_hooks) for m in model.modules())


@pytest.mark.parametrize("kwargs", [{"layers": ["0", "missing"]}, {"layers": ["0"], "processing": "bad"}])
def test_constructor_failure_is_atomic(kwargs):
    """Leave no hooks behind when layer or processing validation fails."""
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    with pytest.raises(ValueError):
        FeatureExtractor(model, **kwargs)
    assert hook_count(model) == 0


def test_registration_exception_cleans_up(monkeypatch):
    """Remove earlier hooks if a later hook registration raises."""
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU())

    def fail(*args, **kwargs):
        """Simulate a hook-registration failure after an earlier successful hook."""
        raise RuntimeError("registration failed")

    monkeypatch.setattr(model[1], "register_forward_hook", fail)
    with pytest.raises(RuntimeError):
        FeatureExtractor(model, layers=["0", "1"])
    assert hook_count(model) == 0


def test_inplace_snapshot_and_forward_independence(device):
    """Keep captured features unchanged by in-place layers and later forwards."""
    model = nn.Sequential(nn.Identity(), nn.ReLU(inplace=True)).to(device)
    collector = FeatureExtractor(model, ["0"])
    try:
        first = collector(input=torch.tensor([[-1.0, 2.0]], device=device))
        assert torch.equal(first["0"], torch.tensor([[-1.0, 2.0]], device=device))
        second = collector(input=torch.tensor([[3.0, -4.0]], device=device))
        assert torch.equal(first["0"], torch.tensor([[-1.0, 2.0]], device=device))
        assert torch.equal(second["0"], torch.tensor([[3.0, -4.0]], device=device))
    finally:
        collector.remove_hooks()
    collector.remove_hooks()
    assert hook_count(model) == 0


@pytest.mark.parametrize("processing", ["none", "flatten", "gap"])
def test_feature_shapes_and_no_grad(device, processing):
    """Check each processing mode's output shape and detached feature snapshots."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1)).to(device)
    extractor = FeatureExtractor(model, ["0"], processing)
    try:
        output = extractor(torch.randn(3, 3, 8, 8, device=device, requires_grad=True))["0"]
        expected = {"none": (3, 4, 8, 8), "flatten": (3, 256), "gap": (3, 4)}[processing]
        assert output.shape == expected
        assert not output.requires_grad
    finally:
        extractor.remove_hooks()


@pytest.mark.parametrize("raise_inside", [False, True])
def test_context_cleanup_restores_mixed_modes(device, raise_inside):
    """Restore every module's training flag and hooks on normal and error exits."""
    model = nn.Sequential(nn.Linear(3, 4), nn.Dropout()).to(device).train()
    model[1].eval()
    states = [m.training for m in model.modules()]
    profiler = ModelProfiler(model, ["0"])
    assert [m.training for m in model.modules()] == states
    try:
        with profiler:
            assert not any(m.training for m in model.modules())
            profiler(torch.randn(4, 3, device=device))
            if raise_inside:
                raise RuntimeError("forward failure")
    except RuntimeError:
        if not raise_inside:
            raise
    assert hook_count(model) == 0
    assert [m.training for m in model.modules()] == states


class KeywordModel(nn.Module):
    """Provide a model with multiple positional or keyword inputs for API tests."""

    def __init__(self):
        """Create the linear feature layer used by the test model."""
        super().__init__()
        self.layer = nn.Linear(3, 4)

    def forward(self, pixels, scale):
        """Project input pixels and scale the model output."""
        return self.layer(pixels) * scale


@pytest.mark.parametrize("input_kind", ["tensor", "tuple", "mapping"])
def test_public_api_and_json_report(device, tmp_path, input_kind, missing_pandas):
    """Profile each input structure and export JSON even when pandas is absent."""
    x = torch.randn(8, 3, device=device)
    if input_kind == "tensor":
        model = nn.Sequential(nn.Linear(3, 4)).to(device)
        inputs, layers = x, ["0"]
    else:
        model = KeywordModel().to(device)
        inputs = (x, 2.0) if input_kind == "tuple" else {"pixels": x, "scale": 2.0}
        layers = ["layer"]
    path = tmp_path / "report.json"
    result = model_profile_under_input_changes(
        model, inputs, inputs, {"layers": layers, "metrics": ["linear", "cosine", "l2"]}, output=path
    )
    assert result.results[0]["linear_" + layers[0]] == pytest.approx(1, abs=1e-10)
    assert json.loads(path.read_text())[0]["l2_" + layers[0]] == 0
    assert hook_count(model) == 0


@pytest.mark.parametrize("kind", ["blur", "brightness"])
def test_actual_kornia_augmentation_and_wrapper(device, kind):
    """Verify deterministic augmentations and every selected layer's report values."""
    x = torch.rand(8, 3, 16, 16, device=device)
    params = {"kernel_size": (5, 5), "sigma": (1.0, 1.0)} if kind == "blur" else {"brightness": (0.8, 0.8)}
    config = [{"name": kind, "params": params}]
    augmenter = AugmentationPipeline(config, mode="individual")
    changed = augmenter(x)[kind]
    assert torch.equal(changed, augmenter(x)[kind])
    expected = kornia.filters.gaussian_blur2d(x, (5, 5), (1.0, 1.0)) if kind == "blur" else (x - 0.2).clamp(0, 1)
    torch.testing.assert_close(changed, expected)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU()).to(device)
    result = model_profile_under_augmentation(
        model,
        {
            "input": x,
            "layers": ["0", "1"],
            "processing": "none",
            "augmentations": config,
            "metrics": ["linear", "gram", "cosine", "l2"],
        },
    )
    assert len(result.results) == 1
    assert all("gram_" + layer in result.results[0] for layer in ["0", "1"])
    assert result.results[0]["tag"] == kind
    assert hook_count(model) == 0


def test_multiple_augmentations_reuse_original(device):
    """Capture the original once and retain independently verified results for every augmentation."""
    model = nn.Sequential(nn.Identity()).to(device)
    inputs = torch.linspace(0.25, 0.75, 2 * 3 * 8 * 8, device=device).reshape(2, 3, 8, 8)
    calls = []

    def count_forward(module, args, output):
        """Record one marker for each model forward without retaining tensors."""
        calls.append(None)

    handle = model.register_forward_hook(count_forward)
    try:
        report = model_profile_under_augmentation(
            model,
            {
                "input": inputs,
                "layers": ["0"],
                "metrics": ["cosine", "l2"],
                "augmentations": [
                    {"name": "rotation", "params": {"degrees": (0.0, 0.0)}},
                    {"name": "brightness", "params": {"brightness": (0.8, 0.8)}},
                ],
            },
        )
    finally:
        handle.remove()
    assert len(calls) == 3  # One original plus two augmented model forwards.
    assert [row["tag"] for row in report.results] == ["rotation", "brightness"]
    assert report.results[0]["l2_0"] == pytest.approx(0, abs=1e-6)
    reference = (
        (
            torch.linalg.vector_norm(torch.full_like(inputs.flatten(1), 0.2), dim=1)
            / (torch.linalg.vector_norm(inputs.flatten(1), dim=1) + 1e-8)
        )
        .mean()
        .item()
    )
    assert report.results[1]["l2_0"] == pytest.approx(reference, rel=1e-5)
    # All comparisons refer to the same detached snapshot, with no duplicate tensor allocation.
    assert report.storage["group_a"][0] is report.storage["group_a"][1]
    assert hook_count(model) == 0


@pytest.mark.parametrize("api", ["inputs", "augmentation"])
def test_single_image_metric_selection(device, api):
    """Require explicit non-CKA metrics for singleton batches and restore model state on failure."""
    model = nn.Sequential(nn.Identity()).to(device).train()
    inputs = torch.rand(1, 3, 8, 8, device=device)
    config = {"input": inputs, "layers": ["0"], "augmentations": [{"name": "brightness"}]}

    def profile():
        """Invoke the selected high-level API with the current metric configuration."""
        if api == "inputs":
            return model_profile_under_input_changes(model, inputs, inputs * 0.8, config)
        return model_profile_under_augmentation(model, config)

    with pytest.raises(ValueError, match=r"at least two samples.*cosine.*l2"):
        profile()
    assert model.training
    assert hook_count(model) == 0
    config["metrics"] = ["cosine", "l2"]
    report = profile()
    assert set(report.results[0]) == {"tag", "cosine_0", "l2_0"}


def test_pandas_report_install_hint(tmp_path, missing_pandas):
    """Name the profiler extra when a CSV report needs an unavailable pandas installation."""
    model = nn.Sequential(nn.Identity())
    inputs = torch.ones(2, 3)
    report = model_profile_under_input_changes(model, inputs, inputs, {"layers": ["0"], "metrics": ["cosine"]})
    with pytest.raises(ImportError, match=r"kornia\[profiler\]"):
        report.save_as_report(tmp_path / "report.csv")


def test_csv_report_with_pandas(tmp_path):
    """Round-trip a CSV report when the optional profiler extra is installed."""
    pd = pytest.importorskip("pandas")
    model = nn.Sequential(nn.Identity())
    inputs = torch.ones(2, 3)
    report = model_profile_under_input_changes(model, inputs, inputs, {"layers": ["0"], "metrics": ["cosine"]})
    path = tmp_path / "report.csv"
    report.save_as_report(path)
    assert pd.read_csv(path).to_dict(orient="records") == report.results


def test_repeated_cuda_forward_has_bounded_retained_memory():
    """Ensure replaced feature snapshots do not retain unbounded CUDA memory."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1)).cuda()
    x = torch.randn(16, 3, 16, 16, device="cuda")
    extractor = FeatureExtractor(model, ["0"])
    try:
        for _ in range(5):
            extractor(x)
        torch.cuda.synchronize()
        initial = torch.cuda.memory_allocated()
        for _ in range(40):
            extractor(x)
        gc.collect()
        torch.cuda.synchronize()
        # Allocated live tensors, not cached allocator reservations.
        assert torch.cuda.memory_allocated() <= initial + 1024 * 1024
    finally:
        extractor.clear()
        extractor.remove_hooks()
    assert hook_count(model) == 0
