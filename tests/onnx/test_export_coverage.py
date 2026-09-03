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
from kornia.core.utils import _torch_inverse_cast
from kornia.geometry.epipolar.numeric import matrix_cofactor_tensor

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

if not hasattr(torch.onnx, "ONNXProgram"):
    # ``torch.onnx.export(..., dynamo=True)`` and the ``ONNXProgram`` it returns (with ``.save()``)
    # arrived together in torch 2.5; older releases only have the TorchScript exporter.
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
    feeds = {inp.name: x.numpy() for inp, x in zip(session.get_inputs(), inputs)}
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
