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

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import onnx  
import onnxruntime as ort 


class TestResizeONNX:

    def test_resize_fixed_size_bilinear(self):
        model = Resize((32, 32), interpolation='bilinear')
        model.eval()

        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            torch_out = model(x)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name

        try:
            torch.onnx.export(
                model,
                x,
                temp_path,
                opset_version=17,
                input_names=['input'],
                output_names=['output'],
                do_constant_folding=True,
                dynamo=False 
            )

            ort_session = ort.InferenceSession(temp_path)
            ort_inputs = {'input': x.numpy()}
            ort_out = ort_session.run(['output'], ort_inputs)[0]

            np.testing.assert_allclose(
                torch_out.numpy(),
                ort_out,
                rtol=1e-4,
                atol=1e-4
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_resize_upscale(self):
        model = Resize((128, 128), interpolation='bilinear')
        model.eval()

        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            torch_out = model(x)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name

        try:
            torch.onnx.export(
                model,
                x,
                temp_path,
                opset_version=17,
                input_names=['input'],
                output_names=['output'],
                dynamo=False
            )

            ort_session = ort.InferenceSession(temp_path)
            ort_out = ort_session.run(['output'], {'input': x.numpy()})[0]

            np.testing.assert_allclose(
                torch_out.numpy(),
                ort_out,
                rtol=1e-4,
                atol=1e-4
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_resize_downscale(self):
        model = Resize((16, 16), interpolation='bilinear')
        model.eval()

        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            torch_out = model(x)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name

        try:
            torch.onnx.export(
                model,
                x,
                temp_path,
                opset_version=17,
                input_names=['input'],
                output_names=['output'],
                dynamo=False
            )

            ort_session = ort.InferenceSession(temp_path)
            ort_out = ort_session.run(['output'], {'input': x.numpy()})[0]

            np.testing.assert_allclose(
                torch_out.numpy(),
                ort_out,
                rtol=1e-4,
                atol=1e-4
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_resize_nearest(self):
        model = Resize((32, 32), interpolation='nearest')
        model.eval()

        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            torch_out = model(x)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name

        try:
            torch.onnx.export(
                model,
                x,
                temp_path,
                opset_version=17,
                input_names=['input'],
                output_names=['output'],
                dynamo=False
            )

            ort_session = ort.InferenceSession(temp_path)
            ort_out = ort_session.run(['output'], {'input': x.numpy()})[0]

            np.testing.assert_allclose(
                torch_out.numpy(),
                ort_out,
                rtol=1e-4,
                atol=1e-4
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_resize_onnx_ops_used(self):
        model = Resize((32, 32))
        model.eval()

        x = torch.randn(1, 3, 64, 64)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name

        try:
            torch.onnx.export(
                model,
                x,
                temp_path,
                opset_version=17,
                dynamo=False
            )
            onnx_model = onnx.load(temp_path)
            ops_used = {node.op_type for node in onnx_model.graph.node}
            assert 'Resize' in ops_used, f"Expected 'Resize' in graph, got: {ops_used}"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)