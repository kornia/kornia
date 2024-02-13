from unittest.mock import PropertyMock, patch

import pytest
import torch

import kornia

from testing.base import BaseTester


class TestImageStitcher(BaseTester):
    @pytest.mark.parametrize("estimator", ["ransac", "vanilla"])
    def test_smoke(self, estimator, device, dtype):
        B, C, H, W = 1, 3, 6, 6
        sample1 = torch.tensor(
            [
                [0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251],
                [0.2071, 0.6297, 0.3653, 0.8513, 0.8549, 0.5509],
                [0.2868, 0.2063, 0.4451, 0.3593, 0.7204, 0.0731],
                [0.9699, 0.1078, 0.8829, 0.4132, 0.7572, 0.6948],
                [0.5209, 0.5932, 0.8797, 0.6286, 0.7653, 0.1132],
                [0.8559, 0.6721, 0.6267, 0.5691, 0.7437, 0.9592],
            ],
            dtype=dtype,
            device=device,
        )
        sample2 = torch.tensor(
            [
                [0.3887, 0.2214, 0.3742, 0.1953, 0.7405, 0.2529],
                [0.2332, 0.9314, 0.9575, 0.5575, 0.4134, 0.4355],
                [0.7369, 0.0331, 0.0914, 0.8994, 0.9936, 0.4703],
                [0.1049, 0.5137, 0.2674, 0.4990, 0.7447, 0.7213],
                [0.4414, 0.5550, 0.6361, 0.1081, 0.3305, 0.5196],
                [0.2147, 0.2816, 0.6679, 0.7878, 0.5070, 0.3055],
            ],
            dtype=dtype,
            device=device,
        )
        sample1 = sample1.expand((B, C, H, W))
        sample2 = sample2.expand((B, C, H, W))
        return_value = {
            "keypoints0": torch.tensor(
                [
                    [0.1546, 0.9391],
                    [0.8077, 0.1051],
                    [0.6768, 0.5596],
                    [0.5092, 0.7195],
                    [0.2856, 0.8889],
                    [0.4342, 0.0203],
                    [0.6701, 0.0585],
                    [0.3828, 0.9038],
                    [0.7301, 0.0762],
                    [0.7864, 0.4490],
                    [0.3509, 0.0756],
                    [0.6782, 0.9297],
                    [0.4132, 0.3664],
                    [0.3134, 0.5039],
                    [0.2073, 0.2552],
                ],
                device=device,
                dtype=dtype,
            ),
            "keypoints1": torch.tensor(
                [
                    [0.2076, 0.2669],
                    [0.9679, 0.8137],
                    [0.9536, 0.8317],
                    [0.3718, 0.2456],
                    [0.3875, 0.8450],
                    [0.7592, 0.1687],
                    [0.5173, 0.6760],
                    [0.9446, 0.4570],
                    [0.6164, 0.1867],
                    [0.4732, 0.1786],
                    [0.4090, 0.8089],
                    [0.9742, 0.8943],
                    [0.5996, 0.7427],
                    [0.7038, 0.9210],
                    [0.6272, 0.0796],
                ],
                device=device,
                dtype=dtype,
            ),
            "confidence": torch.tensor(
                [
                    0.9314,
                    0.5951,
                    0.4187,
                    0.0318,
                    0.1434,
                    0.7952,
                    0.8306,
                    0.7511,
                    0.6407,
                    0.7379,
                    0.4363,
                    0.9220,
                    0.8453,
                    0.5075,
                    0.8141,
                ],
                device=device,
                dtype=dtype,
            ),
            "batch_indexes": torch.zeros((15,), device=device, dtype=dtype),
        }
        with patch(
            "kornia.contrib.ImageStitcher.on_matcher", new_callable=PropertyMock, return_value=lambda x: return_value
        ):
            # NOTE: This will need to download the pretrained weights.
            # To avoid that, we mock as below
            matcher = kornia.feature.LoFTR(None)
            stitcher = kornia.contrib.ImageStitcher(matcher, estimator=estimator).to(device=device, dtype=dtype)
            torch.manual_seed(1)  # issue kornia#2027
            out = stitcher(sample1, sample2)
            assert out.shape[:-1] == torch.Size([1, 3, 6])
            assert out.shape[-1] <= 12

    @pytest.mark.slow
    def test_exception(self, device, dtype):
        B, C, H, W = 1, 3, 224, 224
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        sample2 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        # NOTE: This will need to download the pretrained weights.
        matcher = kornia.feature.LoFTR(None)

        with pytest.raises(NotImplementedError):
            stitcher = kornia.contrib.ImageStitcher(matcher, estimator="random").to(device=device, dtype=dtype)

        stitcher = kornia.contrib.ImageStitcher(matcher).to(device=device, dtype=dtype)
        with pytest.raises(RuntimeError):
            stitcher(sample1, sample2)
