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

from kornia.geometry.transform import Resize

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def test_resize_dynamo_with_binding():
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


def test_resize_upscale_dynamo():
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


def test_resize_downscale_dynamo():
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


def test_resize_nearest_dynamo():
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
