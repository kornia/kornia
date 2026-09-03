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

"""Dynamo ONNX export smoke tests for operators whose graph capture needed an export-time path.

Each case pins one lowering fix (closed-form inverse, ``sort`` instead of ``median.dim``, ``amax``
instead of ``AdaptiveMaxPool``, branch-free ``where`` selection, ...). The exported graph is run
under onnxruntime and compared with eager execution.
"""

import io

import numpy as np
import pytest
import torch
from torch import nn

import kornia
from kornia.core._compat import torch_version_lt
from kornia.core.utils import _torch_inverse_cast
from kornia.geometry.epipolar.numeric import matrix_cofactor_tensor

# Every tensor here is created device-free and fed to onnxruntime as numpy: the work is CPU-bound
# whatever ``--device`` says, so the accelerator legs must not repeat it.
pytestmark = pytest.mark.device_agnostic

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")  # the dynamo exporter hard-requires it; the ``docs`` extra lacks it

if torch_version_lt(2, 5, 0):
    # ``torch.onnx.export(..., dynamo=True)`` only reaches the current exporter -- and returns the
    # ``ONNXProgram`` with ``.save()`` this file needs -- from torch 2.5. On 2.1-2.3 the ``dynamo=``
    # keyword does not exist at all (TypeError); 2.4 accepts it but routes to the retired
    # ``dynamo_export``, which the onnxscript releases we resolve no longer serve.
    # ``hasattr(torch.onnx, "ONNXProgram")`` is not a usable probe: that class has existed since 2.2
    # as ``dynamo_export``'s return type, which is why the previous guard did not skip.
    # The floor is deliberately 2.5 rather than the 2.6 that ``tests/filters/test_gaussian.py``
    # requires for its smoke test: these cases pin the export-time code paths on the oldest
    # exporter-capable release, and torch 2.5.1 is a blocking PR leg, so a regression there must
    # fail rather than skip.
    pytest.skip("the dynamo ONNX exporter needs torch >= 2.5", allow_module_level=True)


class _Fn(nn.Module):
    def __init__(self, fn, **kwargs):
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs

    def forward(self, *args):
        return self.fn(*args, **self.kwargs)


def _export_and_run(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[list, list]:
    module.eval()
    with torch.no_grad():
        eager = module(*inputs)
    eager = list(eager) if isinstance(eager, (tuple, list)) else [eager]

    buf = io.BytesIO()
    with torch.no_grad():
        program = torch.onnx.export(module, inputs, dynamo=True, opset_version=18, verbose=False)
    program.save(buf)
    model = onnx.load_from_string(buf.getvalue())
    onnx.checker.check_model(model)

    session = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    # An input the exporter folded into the graph would shorten ``get_inputs()`` and shift the rest
    # into the wrong slots; the feed must match one to one.
    assert len(session.get_inputs()) == len(inputs), [i.name for i in session.get_inputs()]
    feeds = {inp.name: x.numpy() for inp, x in zip(session.get_inputs(), inputs, strict=True)}
    return eager, session.run(None, feeds)


def _assert_same(eager, ort_out, rtol=1e-4, atol=1e-4):
    assert len(eager) == len(ort_out)
    for e, o in zip(eager, ort_out):
        np.testing.assert_allclose(e.numpy(), o, rtol=rtol, atol=atol)


def _cases():
    torch.manual_seed(0)
    img = torch.rand(1, 3, 16, 16)
    gray = torch.rand(2, 1, 12, 12)
    return [
        pytest.param(
            _Fn(_torch_inverse_cast), (torch.eye(4).expand(2, 4, 4) + 0.1 * torch.rand(2, 4, 4),), id="inv4x4"
        ),
        pytest.param(
            _Fn(kornia.geometry.transform.warp_perspective, dsize=(16, 16), align_corners=True),
            (img, torch.eye(3)[None] + 0.01 * torch.rand(1, 3, 3)),
            id="warp_perspective",
        ),
        pytest.param(kornia.filters.MedianBlur((3, 3)), (img,), id="MedianBlur"),
        pytest.param(_Fn(kornia.contrib.distance_transform), ((gray > 0.7).float(),), id="distance_transform"),
        pytest.param(
            kornia.losses.HausdorffERLoss(),
            (torch.rand(2, 3, 12, 12), torch.randint(0, 3, (2, 1, 12, 12))),
            id="hausdorff",
        ),
        pytest.param(_Fn(kornia.feature.match_nn), (torch.rand(8, 32), torch.rand(10, 32)), id="match_nn"),
        pytest.param(
            _Fn(kornia.geometry.solvers.solve_quadratic),
            (torch.tensor([[1.0, -3.0, 2.0], [1.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),),
            id="solve_quadratic",
        ),
        pytest.param(_Fn(matrix_cofactor_tensor), (torch.rand(2, 3, 3),), id="cofactor"),
        pytest.param(
            _Fn(kornia.enhance.normalize),
            (img, torch.tensor([0.4, 0.5, 0.6]), torch.tensor([0.2, 0.2, 0.3])),
            id="normalize",
        ),
        pytest.param(kornia.augmentation.LongestMaxSize(24), (img,), id="LongestMaxSize"),
        pytest.param(
            _Fn(kornia.geometry.calibration.distort_points),
            (torch.rand(1, 5, 2) * 8, torch.eye(3)[None] * 4, torch.rand(1, 5) * 0.1),
            id="distort_points",
        ),
        pytest.param(_Fn(lambda v: kornia.geometry.liegroup.Se3.exp(v).matrix()), (torch.rand(2, 6),), id="Se3.exp"),
    ]


@pytest.mark.parametrize("module,inputs", _cases())
def test_onnx_export_matches_eager(module, inputs):
    eager, ort_out = _export_and_run(module, inputs)
    _assert_same(eager, ort_out)


def test_boxes_to_mask_export():
    boxes = kornia.geometry.boxes.Boxes(torch.tensor([[[[1.0, 1.0], [5.0, 1.0], [5.0, 4.0], [1.0, 4.0]]]]))

    class M(nn.Module):
        def forward(self, quads):
            return kornia.geometry.boxes.Boxes(quads).to_mask(8, 8)

    eager, ort_out = _export_and_run(M(), (boxes.data,))
    _assert_same(eager, ort_out)
