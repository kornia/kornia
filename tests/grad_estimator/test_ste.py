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

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

import kornia.augmentation as K
from kornia.grad_estimator import STEFunction, StraightThroughEstimator


class TestSTE:
    def test_smoke(self):
        StraightThroughEstimator(K.Normalize(0.5, 0.5))

    def test_function(self, device, dtype):
        data = torch.randn(4, requires_grad=True, device=device, dtype=dtype)
        output = torch.sign(data)
        loss = output.mean()
        loss.backward()
        assert_close(data.grad, torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype))

        out_est = STEFunction.apply(data, output, F.hardtanh)
        loss = out_est.mean()
        loss.backward()
        assert_close(data.grad, torch.tensor([0.2500, 0.2500, 0.2500, 0.2500], device=device, dtype=dtype))

        out_est = STEFunction.apply(data, output, None)
        loss = out_est.mean()
        loss.backward()
        assert_close(data.grad, torch.tensor([0.5000, 0.5000, 0.5000, 0.5000], device=device, dtype=dtype))

    def test_module(self, device, dtype):
        data = torch.randn(1, 1, 4, 4, requires_grad=True, device=device, dtype=dtype)
        estimator = StraightThroughEstimator(K.RandomPosterize(3, p=1.0), grad_fn=F.hardtanh)
        out = estimator(data)
        loss = out.mean()
        loss.backward()
        o = torch.tensor(
            [
                [
                    [
                        [0.0625, 0.0625, 0.0625, 0.0625],
                        [0.0625, 0.0625, 0.0625, 0.0625],
                        [0.0625, 0.0625, 0.0625, 0.0625],
                        [0.0625, 0.0625, 0.0625, 0.0625],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(data.grad, o)

    @pytest.mark.skip("Function.apply is not supported in Torchscript rightnow.")
    def test_jit(self, device, dtype):
        inputs = torch.rand(2, 3, 30, 30, dtype=dtype, device=device)

        op = StraightThroughEstimator(torch.nn.MaxPool2d(3), grad_fn=None)
        op_script = torch.jit.script(op)
        actual = op_script(inputs)
        expected = op(inputs)
        assert_close(actual, expected)

    @pytest.mark.skip("Function is not supported to export to onnx rightnow.")
    def test_onnx(self, device, dtype):
        inputs = torch.rand(2, 3, 30, 30, dtype=dtype, device=device)
        model = StraightThroughEstimator(torch.nn.PixelShuffle(1), grad_fn=None)

        input_names = ["input"]
        output_names = ["output1"]

        torch.onnx.export(model, inputs, "t.onnx", verbose=True, input_names=input_names, output_names=output_names)
