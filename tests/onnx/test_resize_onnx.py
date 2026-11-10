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

import os
import tempfile

import numpy as np
import pytest
import torch

from kornia.core._compat import torch_version_lt
from kornia.geometry.transform import Resize

# Every test in this module must stay device-free: the mark deselects the whole file on non-CPU devices.
pytestmark = pytest.mark.device_agnostic

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

if torch_version_lt(2, 5, 0):
    # Same floor as ``tests/onnx/test_export_coverage.py``: ``dynamo=True`` reaches the current
    # exporter from torch 2.5 (no such keyword on 2.1-2.3, the retired ``dynamo_export`` on 2.4).
    # These tests used to carry ``dynamo`` in their names, which the conftest deselects whenever
    # ``KORNIA_TEST_OPTIMIZER`` is unset, so no CI job ever collected them; they are named for
    # what they check instead.
    pytest.skip("the dynamo ONNX exporter needs torch >= 2.5", allow_module_level=True)


def test_resize_export_binding():
    model = Resize((32, 32), interpolation="bilinear")
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch_out = model(x)

    fd, temp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)

    try:
        torch.onnx.export(model, x, temp_path, dynamo=True, opset_version=18)

        ort_session = ort.InferenceSession(temp_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        binding = ort_session.io_binding()
        binding.bind_cpu_input(input_name, x.numpy())
        binding.bind_output(output_name)

        ort_session.run_with_iobinding(binding)
        ort_out = binding.copy_outputs_to_cpu()[0]

        np.testing.assert_allclose(torch_out.numpy(), ort_out, rtol=1e-4, atol=1e-4)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_resize_export_upscale():
    """Test upscaling with dynamo."""
    model = Resize((128, 128))
    model.eval()
    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        torch_out = model(x)

    fd, temp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)

    try:
        torch.onnx.export(model, x, temp_path, dynamo=True, opset_version=18)

        ort_session = ort.InferenceSession(temp_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        binding = ort_session.io_binding()
        binding.bind_cpu_input(input_name, x.numpy())
        binding.bind_output(output_name)
        ort_session.run_with_iobinding(binding)
        ort_out = binding.copy_outputs_to_cpu()[0]

        np.testing.assert_allclose(torch_out.numpy(), ort_out, rtol=1e-4, atol=1e-4)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_resize_export_downscale():
    """Test downscaling with dynamo."""
    model = Resize((16, 16))
    model.eval()
    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        torch_out = model(x)

    fd, temp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)

    try:
        torch.onnx.export(model, x, temp_path, dynamo=True, opset_version=18)

        ort_session = ort.InferenceSession(temp_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        binding = ort_session.io_binding()
        binding.bind_cpu_input(input_name, x.numpy())
        binding.bind_output(output_name)
        ort_session.run_with_iobinding(binding)
        ort_out = binding.copy_outputs_to_cpu()[0]

        np.testing.assert_allclose(torch_out.numpy(), ort_out, rtol=1e-4, atol=1e-4)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_resize_export_nearest():
    """Test nearest neighbor interpolation with dynamo."""
    model = Resize((32, 32), interpolation="nearest")
    model.eval()
    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        torch_out = model(x)

    fd, temp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)

    try:
        torch.onnx.export(model, x, temp_path, dynamo=True, opset_version=18)

        ort_session = ort.InferenceSession(temp_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        binding = ort_session.io_binding()
        binding.bind_cpu_input(input_name, x.numpy())
        binding.bind_output(output_name)
        ort_session.run_with_iobinding(binding)
        ort_out = binding.copy_outputs_to_cpu()[0]

        np.testing.assert_allclose(torch_out.numpy(), ort_out, rtol=1e-4, atol=1e-4)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
