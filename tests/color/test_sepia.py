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

import kornia

from testing.base import BaseTester


class TestSepia(BaseTester):
    def test_smoke(self, device, dtype):
        input_tensor = torch.tensor(
            [[[0.1, 1.0], [0.2, 0.1]], [[0.1, 0.8], [0.2, 0.5]], [[0.1, 0.3], [0.2, 0.8]]], device=device, dtype=dtype
        )

        # With rescale
        expected_tensor = torch.tensor(
            [[[0.1269, 1.0], [0.2537, 0.5400]], [[0.1269, 1.0], [0.2537, 0.5403]], [[0.1269, 1.0], [0.2538, 0.5403]]],
            device=device,
            dtype=dtype,
        )
        actual = kornia.color.sepia(input_tensor, rescale=True)

        assert actual.shape[:] == (3, 2, 2)
        self.assert_close(actual, expected_tensor, rtol=1e-2, atol=1e-2)

        # Without rescale
        expected_tensor = torch.tensor(
            [
                [[0.1351, 1.0649], [0.2702, 0.5750]],
                [[0.1203, 0.9482], [0.2406, 0.5123]],
                [[0.0937, 0.7385], [0.1874, 0.3990]],
            ],
            device=device,
            dtype=dtype,
        )

        actual = kornia.color.sepia(input_tensor, rescale=False)
        assert actual.shape[:] == (3, 2, 2)
        self.assert_close(actual, expected_tensor, rtol=1e-2, atol=1e-2)

    def test_exception(self, device, dtype):
        from kornia.core.exceptions import ShapeError
        
        with pytest.raises(ShapeError):
            kornia.color.sepia(torch.rand(size=(4, 1, 1), dtype=dtype, device=device))

    @pytest.mark.parametrize("batch_shape", [(1, 3, 8, 15), (2, 3, 11, 7), (3, 8, 15)])
    def test_cardinality(self, batch_shape, device, dtype):
        input_tensor = torch.rand(batch_shape, device=device, dtype=dtype)
        actual = kornia.color.sepia(input_tensor)
        assert actual.shape == batch_shape

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = kornia.color.sepia(inp)

        assert inp.is_contiguous() is False
        assert actual.is_contiguous()
        self.assert_close(actual, actual)

    def test_gradcheck(self, device):
        # test parameters
        batch_shape = (1, 3, 5, 5)

        # evaluate function gradient
        input_tensor = torch.rand(batch_shape, device=device, dtype=torch.float64)
        self.gradcheck(kornia.color.sepia, (input_tensor,))

    def test_jit(self, device, dtype):
        op = kornia.color.sepia
        op_script = torch.jit.script(op)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(op(img), op_script(img))

    def test_module(self, device, dtype):
        op = kornia.color.sepia
        op_module = kornia.color.Sepia()

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(op(img), op_module(img))
