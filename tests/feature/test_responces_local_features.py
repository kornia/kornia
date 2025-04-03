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

import torch

import kornia

from testing.base import BaseTester


class TestCornerHarris(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4, device=device)
        harris = kornia.feature.CornerHarris(k=0.04).to(device)
        assert harris(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 6, 4, 4, device=device)
        harris = kornia.feature.CornerHarris(k=0.04).to(device)
        assert harris(inp).shape == (2, 6, 4, 4)

    def test_corners(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).float()

        expected = torch.tensor(
            [
                [
                    [
                        [0.0042, 0.0054, 0.0035, 0.0006, 0.0035, 0.0054, 0.0042],
                        [0.0054, 0.0068, 0.0046, 0.0014, 0.0046, 0.0068, 0.0054],
                        [0.0035, 0.0046, 0.0034, 0.0014, 0.0034, 0.0046, 0.0035],
                        [0.0006, 0.0014, 0.0014, 0.0006, 0.0014, 0.0014, 0.0006],
                        [0.0035, 0.0046, 0.0034, 0.0014, 0.0034, 0.0046, 0.0035],
                        [0.0054, 0.0068, 0.0046, 0.0014, 0.0046, 0.0068, 0.0054],
                        [0.0042, 0.0054, 0.0035, 0.0006, 0.0035, 0.0054, 0.0042],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).float()
        harris = kornia.feature.CornerHarris(k=0.04).to(device)
        scores = harris(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-3)

    def test_corners_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = (
            torch.tensor(
                [
                    [
                        [0.0415, 0.0541, 0.0346, 0.0058, 0.0346, 0.0541, 0.0415],
                        [0.0541, 0.0678, 0.0457, 0.0145, 0.0457, 0.0678, 0.0541],
                        [0.0346, 0.0457, 0.0335, 0.0139, 0.0335, 0.0457, 0.0346],
                        [0.0058, 0.0145, 0.0139, 0.0064, 0.0139, 0.0145, 0.0058],
                        [0.0346, 0.0457, 0.0335, 0.0139, 0.0335, 0.0457, 0.0346],
                        [0.0541, 0.0678, 0.0457, 0.0145, 0.0457, 0.0678, 0.0541],
                        [0.0415, 0.0541, 0.0346, 0.0058, 0.0346, 0.0541, 0.0415],
                    ],
                    [
                        [0.0415, 0.0547, 0.0447, 0.0440, 0.0490, 0.0182, 0.0053],
                        [0.0547, 0.0688, 0.0557, 0.0549, 0.0610, 0.0229, 0.0066],
                        [0.0447, 0.0557, 0.0444, 0.0437, 0.0489, 0.0168, 0.0035],
                        [0.0440, 0.0549, 0.0437, 0.0431, 0.0481, 0.0166, 0.0034],
                        [0.0490, 0.0610, 0.0489, 0.0481, 0.0541, 0.0205, 0.0060],
                        [0.0182, 0.0229, 0.0168, 0.0166, 0.0205, 0.0081, 0.0025],
                        [0.0053, 0.0066, 0.0035, 0.0034, 0.0060, 0.0025, 0.0008],
                    ],
                ],
                device=device,
                dtype=dtype,
            ).repeat(2, 1, 1, 1)
            / 10.0
        )
        scores = kornia.feature.harris_response(inp, k=0.04)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        k = 0.04
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.harris_response, (img, k), nondet_tol=1e-4)


class TestCornerGFTT(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4, device=device)
        shi_tomasi = kornia.feature.CornerGFTT().to(device)
        assert shi_tomasi(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 6, 4, 4, device=device)
        shi_tomasi = kornia.feature.CornerGFTT().to(device)
        assert shi_tomasi(inp).shape == (2, 6, 4, 4)

    def test_corners(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).float()

        expected = torch.tensor(
            [
                [
                    [
                        [0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379],
                        [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                        [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                        [0.0121, 0.0168, 0.0245, 0.0276, 0.0245, 0.0168, 0.0121],
                        [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                        [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                        [0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).float()
        shi_tomasi = kornia.feature.CornerGFTT().to(device)
        scores = shi_tomasi(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-3)

    def test_corners_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [
                [
                    [0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379],
                    [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                    [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                    [0.0121, 0.0168, 0.0245, 0.0276, 0.0245, 0.0168, 0.0121],
                    [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                    [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                    [0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379],
                ],
                [
                    [0.0379, 0.0462, 0.0349, 0.0345, 0.0443, 0.0248, 0.0112],
                    [0.0462, 0.0608, 0.0488, 0.0483, 0.0581, 0.0274, 0.0119],
                    [0.0349, 0.0488, 0.0669, 0.0664, 0.0460, 0.0191, 0.0084],
                    [0.0345, 0.0483, 0.0664, 0.0660, 0.0455, 0.0189, 0.0083],
                    [0.0443, 0.0581, 0.0460, 0.0455, 0.0555, 0.0262, 0.0114],
                    [0.0248, 0.0274, 0.0191, 0.0189, 0.0262, 0.0172, 0.0084],
                    [0.0112, 0.0119, 0.0084, 0.0083, 0.0114, 0.0084, 0.0046],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        shi_tomasi = kornia.feature.CornerGFTT().to(device)
        scores = shi_tomasi(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.gftt_response, (img), nondet_tol=1e-4)


class TestBlobHessian(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4, device=device)
        shi_tomasi = kornia.feature.BlobHessian().to(device)
        assert shi_tomasi(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 6, 4, 4, device=device)
        shi_tomasi = kornia.feature.BlobHessian().to(device)
        assert shi_tomasi(inp).shape == (2, 6, 4, 4)

    def test_blobs_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [
                [
                    [-0.0564, -0.0759, -0.0342, -0.0759, -0.0564, -0.0057, 0.0000],
                    [-0.0759, -0.0330, 0.0752, -0.0330, -0.0759, -0.0096, 0.0000],
                    [-0.0342, 0.0752, 0.1914, 0.0752, -0.0342, -0.0068, 0.0000],
                    [-0.0759, -0.0330, 0.0752, -0.0330, -0.0759, -0.0096, 0.0000],
                    [-0.0564, -0.0759, -0.0342, -0.0759, -0.0564, -0.0057, 0.0000],
                    [-0.0057, -0.0096, -0.0068, -0.0096, -0.0057, -0.0005, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                ],
                [
                    [-0.0564, -0.0522, -0.0522, -0.0564, -0.0057, 0.0000, 0.0000],
                    [-0.0522, 0.0688, 0.0688, -0.0123, 0.0033, -0.0057, -0.0005],
                    [-0.0522, 0.0688, -0.0755, -0.1111, -0.0123, -0.0564, -0.0057],
                    [-0.0564, -0.0123, -0.1111, -0.0755, 0.0688, -0.0522, -0.0080],
                    [-0.0057, 0.0033, -0.0123, 0.0688, 0.0688, -0.0522, -0.0080],
                    [0.0000, -0.0057, -0.0564, -0.0522, -0.0522, -0.0564, -0.0057],
                    [0.0000, -0.0005, -0.0057, -0.0080, -0.0080, -0.0057, -0.0005],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        shi_tomasi = kornia.feature.BlobHessian().to(device)
        scores = shi_tomasi(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.hessian_response, (img), nondet_tol=1e-4)


class TestBlobDoGSingle(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 9, 9, device=device)
        shi_tomasi = kornia.feature.BlobDoGSingle().to(device)
        assert shi_tomasi(inp).shape == (1, 3, 9, 9)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 6, 9, 9, device=device)
        shi_tomasi = kornia.feature.BlobHessian().to(device)
        assert shi_tomasi(inp).shape == (2, 6, 9, 9)

    def test_blobs_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).expand(2, 2, 7, 7)
        expected = torch.tensor(
            [
                [
                    [
                        [0.1640, 0.0811, -0.0148, 0.0040, 0.0850, 0.0962, 0.0754],
                        [0.0811, -0.0249, -0.1416, -0.0950, 0.0437, 0.0894, 0.0754],
                        [-0.0148, -0.1416, -0.2758, -0.2005, -0.0044, 0.0777, 0.0718],
                        [0.0040, -0.0950, -0.2005, -0.1445, 0.0045, 0.0648, 0.0586],
                        [0.0850, 0.0437, -0.0044, 0.0045, 0.0441, 0.0488, 0.0382],
                        [0.0962, 0.0894, 0.0777, 0.0648, 0.0488, 0.0295, 0.0197],
                        [0.0754, 0.0754, 0.0718, 0.0586, 0.0382, 0.0197, 0.0124],
                    ],
                    [
                        [0.0689, -0.0056, -0.0254, 0.0802, 0.1118, 0.0708, 0.0491],
                        [-0.0056, -0.0895, -0.1031, 0.0362, 0.0986, 0.0771, 0.0621],
                        [-0.0254, -0.1031, -0.1423, -0.0651, 0.0039, 0.0567, 0.0794],
                        [0.0802, 0.0362, -0.0651, -0.1795, -0.1617, -0.0048, 0.0771],
                        [0.1118, 0.0986, 0.0039, -0.1617, -0.1706, -0.0095, 0.0761],
                        [0.0708, 0.0771, 0.0567, -0.0048, -0.0095, 0.0521, 0.0836],
                        [0.0491, 0.0621, 0.0794, 0.0771, 0.0761, 0.0836, 0.0858],
                    ],
                ],
                [
                    [
                        [0.1640, 0.0811, -0.0148, 0.0040, 0.0850, 0.0962, 0.0754],
                        [0.0811, -0.0249, -0.1416, -0.0950, 0.0437, 0.0894, 0.0754],
                        [-0.0148, -0.1416, -0.2758, -0.2005, -0.0044, 0.0777, 0.0718],
                        [0.0040, -0.0950, -0.2005, -0.1445, 0.0045, 0.0648, 0.0586],
                        [0.0850, 0.0437, -0.0044, 0.0045, 0.0441, 0.0488, 0.0382],
                        [0.0962, 0.0894, 0.0777, 0.0648, 0.0488, 0.0295, 0.0197],
                        [0.0754, 0.0754, 0.0718, 0.0586, 0.0382, 0.0197, 0.0124],
                    ],
                    [
                        [0.0689, -0.0056, -0.0254, 0.0802, 0.1118, 0.0708, 0.0491],
                        [-0.0056, -0.0895, -0.1031, 0.0362, 0.0986, 0.0771, 0.0621],
                        [-0.0254, -0.1031, -0.1423, -0.0651, 0.0039, 0.0567, 0.0794],
                        [0.0802, 0.0362, -0.0651, -0.1795, -0.1617, -0.0048, 0.0771],
                        [0.1118, 0.0986, 0.0039, -0.1617, -0.1706, -0.0095, 0.0761],
                        [0.0708, 0.0771, 0.0567, -0.0048, -0.0095, 0.0521, 0.0836],
                        [0.0491, 0.0621, 0.0794, 0.0771, 0.0761, 0.0836, 0.0858],
                    ],
                ],
            ],
            device=device,
            dtype=dtype,
        ).expand(2, 2, 7, 7)
        det = kornia.feature.BlobDoGSingle(1.0, 1.6).to(device)
        scores = det(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 9, 11
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.dog_response_single, (img), nondet_tol=1e-4)
