import torch

import kornia

from testing.base import BaseTester


class TestNMS2d(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((3, 3)).to(device)
        assert nms(inp).shape == inp.shape

    def test_shape_batch(self, device):
        inp = torch.ones(4, 3, 4, 4, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((3, 3)).to(device)
        assert nms(inp).shape == inp.shape

    def test_nms(self, device):
        inp = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.7, 1.1, 0.0, 1.0, 2.0, 0.0],
                        [0.0, 0.8, 1.0, 0.0, 1.0, 1.0, 0.0],
                    ]
                ]
            ],
            device=device,
        ).float()

        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0, 0, 0.0, 0, 0.0, 0.0],
                        [0.0, 0, 1.1, 0.0, 0.0, 2.0, 0.0],
                        [0.0, 0, 0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
        ).float()
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((3, 3)).to(device)
        scores = nms(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.nms2d, (img, (3, 3)), nondet_tol=1e-4)


class TestNMS3d(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 1, 3, 4, 4, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression3d((3, 3, 3)).to(device)
        assert nms(inp).shape == inp.shape

    def test_shape_batch(self, device):
        inp = torch.ones(4, 1, 3, 4, 4, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression3d((3, 3, 3)).to(device)
        assert nms(inp).shape == inp.shape

    def test_nms(self, device):
        inp = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 2.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ]
        ).to(device)

        expected = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 2.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ]
        ).to(device)
        nms = kornia.geometry.subpix.NonMaximaSuppression3d((3, 3, 3)).to(device)
        scores = nms(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, depth, height, width = 1, 1, 4, 5, 4
        img = torch.rand(batch_size, channels, depth, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.nms3d, (img, (3, 3, 3)), nondet_tol=1e-4)
