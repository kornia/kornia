import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestCornerHarris:
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
        assert_close(scores, expected, atol=1e-4, rtol=1e-3)

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
        assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        k = 0.04
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(
            kornia.feature.harris_response, (img, k), raise_exception=True, nondet_tol=1e-4, fast_mode=True
        )


class TestCornerGFTT:
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
        assert_close(scores, expected, atol=1e-4, rtol=1e-3)

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
        assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.gftt_response, (img), raise_exception=True, nondet_tol=1e-4, fast_mode=True)


class TestBlobHessian:
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
        assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.hessian_response, (img), raise_exception=True, nondet_tol=1e-4, fast_mode=True)


class TestBlobDoGSingle:
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
                        [0.1684, 0.0856, -0.0118, 0.0059, 0.0842, 0.0956, 0.0734],
                        [0.0856, -0.0203, -0.1385, -0.0931, 0.0428, 0.0888, 0.0733],
                        [-0.0118, -0.1385, -0.2741, -0.1998, -0.0059, 0.0768, 0.0695],
                        [0.0059, -0.0931, -0.1998, -0.1443, 0.0030, 0.0639, 0.0567],
                        [0.0842, 0.0428, -0.0059, 0.0030, 0.0421, 0.0478, 0.0367],
                        [0.0956, 0.0888, 0.0768, 0.0639, 0.0478, 0.0289, 0.0189],
                        [0.0734, 0.0733, 0.0695, 0.0567, 0.0367, 0.0189, 0.0117],
                    ],
                    [
                        [0.0725, -0.0023, -0.0226, 0.0810, 0.1114, 0.0699, 0.0471],
                        [-0.0023, -0.0868, -0.1010, 0.0361, 0.0974, 0.0758, 0.0598],
                        [-0.0226, -0.1010, -0.1408, -0.0652, 0.0030, 0.0559, 0.0777],
                        [0.0810, 0.0361, -0.0652, -0.1802, -0.1625, -0.0049, 0.0764],
                        [0.1114, 0.0974, 0.0030, -0.1625, -0.1710, -0.0092, 0.0761],
                        [0.0699, 0.0758, 0.0559, -0.0049, -0.0092, 0.0531, 0.0844],
                        [0.0471, 0.0598, 0.0777, 0.0764, 0.0761, 0.0844, 0.0865],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        ).expand(2, 2, 7, 7)
        det = kornia.feature.BlobDoGSingle(1.0, 1.6).to(device)
        scores = det(inp)
        assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 9, 11
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(
            kornia.feature.dog_response_single, (img), raise_exception=True, nondet_tol=1e-4, fast_mode=True
        )
