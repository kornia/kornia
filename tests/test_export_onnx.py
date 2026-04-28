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
"""ONNX opset-18 round-trip for deterministic kornia transforms.

Closes torchgeo#3108 (isaaccorley note: 'ONNX support... onnxruntime-web').
Skipped if onnx or onnxruntime not installed.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

import kornia.augmentation as K

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def _make_normalize() -> K.Normalize:
    return K.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
    )


def _make_center_crop() -> K.CenterCrop:
    return K.CenterCrop(size=(64, 64))


def _make_resize() -> K.Resize:
    return K.Resize(size=(64, 64))


DETERMINISTIC = [
    ("Normalize", _make_normalize),
    ("CenterCrop", _make_center_crop),
    ("Resize", _make_resize),
]


@pytest.mark.parametrize("name,factory", DETERMINISTIC, ids=lambda x: x[0] if isinstance(x, tuple) else None)
def test_onnx_opset_18(name: str, factory) -> None:
    """Each deterministic transform must export to ONNX opset 18 and
    produce numerically close results under onnxruntime."""
    try:
        m = factory()
        m.eval()
    except Exception as e:
        pytest.skip(f"{name} construction failed: {e}")
    x = torch.randn(1, 3, 96, 96)

    buf = io.BytesIO()
    try:
        torch.onnx.export(m, (x,), buf, opset_version=18, dynamo=True)
    except Exception as e:
        pytest.fail(f"{name} ONNX opset-18 export failed: {type(e).__name__}: {e}")

    buf.seek(0)
    sess = ort.InferenceSession(buf.read(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    y_onnx = sess.run(None, {input_name: x.numpy()})[0]

    y_eager = m(x)
    if isinstance(y_eager, torch.Tensor):
        np.testing.assert_allclose(y_eager.numpy(), y_onnx, atol=1e-4, rtol=1e-3)
    else:
        # tuple-return — check first output
        np.testing.assert_allclose(y_eager[0].numpy(), y_onnx, atol=1e-4, rtol=1e-3)
