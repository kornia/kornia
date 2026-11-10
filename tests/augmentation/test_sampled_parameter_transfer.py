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

import kornia.augmentation as K

from testing.base import BaseTester

# Every sampler draws 0.25 (torch.rand is fixed in the test), so each parameter has a known value. Each case lists the
# parameters for one sample; every sample of the batch has the same value. The values follow from the ranges below.
CASES = [
    pytest.param(
        lambda: K.RandomShear((-20.0, 20.0, -10.0, 10.0), p=1.0),
        (2, 3, 8, 9),
        # -20 + 0.25 * 40 and -10 + 0.25 * 20 degrees.
        {"shear_x": -10.0, "shear_y": -5.0},
        id="shear",
    ),
    pytest.param(
        lambda: K.RandomTranslate(0.2, 0.2, p=1.0),
        (2, 3, 8, 9),
        # (-0.2 + 0.25 * 0.4) of the 9 x 8 image.
        {"translate_x": -0.9, "translate_y": -0.8},
        id="translate",
    ),
    pytest.param(
        lambda: K.RandomCrop((4, 5), p=1.0),
        (2, 3, 8, 9),
        # A 5 x 4 crop of a 9 x 8 image starts at floor(0.25 * 5) = 1 in x and in y.
        {"src": [[1.0, 1.0], [5.0, 1.0], [5.0, 4.0], [1.0, 4.0]]},
        id="crop",
    ),
    pytest.param(
        lambda: K.RandomResizedCrop((6, 7), p=1.0),
        (2, 3, 8, 9),
        # Area (0.08 + 0.25 * 0.92) * 72 = 22.32 and aspect ratio (3 / 4) ** 0.5 give a 4 x 5 crop (rounded), which
        # starts at floor(0.25 * 6) = 1 in x and floor(0.25 * 4) = 1 in y.
        {"src": [[1.0, 1.0], [4.0, 1.0], [4.0, 5.0], [1.0, 5.0]]},
        id="resized-crop",
    ),
    pytest.param(
        lambda: K.RandomCrop3D((3, 4, 5), p=1.0),
        (1, 1, 7, 8, 9),
        # A 5 x 4 x 3 crop of a 9 x 8 x 7 volume starts at floor(0.25 * 5) = 1 on each axis.
        {
            "src": [
                [1.0, 1.0, 1.0],
                [5.0, 1.0, 1.0],
                [5.0, 4.0, 1.0],
                [1.0, 4.0, 1.0],
                [1.0, 1.0, 3.0],
                [5.0, 1.0, 3.0],
                [5.0, 4.0, 3.0],
                [1.0, 4.0, 3.0],
            ]
        },
        id="crop3d",
    ),
    pytest.param(
        lambda: K.RandomPerspective3D(0.5, p=1.0),
        (1, 1, 5, 8, 9),
        # distortion_scale=0.5 on a 9 x 8 x 5 volume: the largest offset is (2.25, 2.0, 1.25), and each corner moves
        # inward by a quarter of it.
        {
            "end_points": [
                [0.5625, 0.5, 0.3125],
                [7.4375, 0.5, 0.3125],
                [7.4375, 6.5, 0.3125],
                [0.5625, 6.5, 0.3125],
                [0.5625, 0.5, 3.6875],
                [7.4375, 0.5, 3.6875],
                [7.4375, 6.5, 3.6875],
                [0.5625, 6.5, 3.6875],
            ]
        },
        id="perspective3d",
    ),
    pytest.param(
        lambda: K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0),
        (2, 3, 8, 9),
        # 0.8 + 0.25 * 0.4 for brightness, contrast and saturation, and -0.1 + 0.25 * 0.2 for hue.
        {"brightness_factor": 0.9, "contrast_factor": 0.9, "saturation_factor": 0.9, "hue_factor": -0.05},
        id="color-jiggle",
    ),
    pytest.param(
        lambda: K.RandomAffine3D(30.0, translate=(0.1, 0.1, 0.1), scale=(0.8, 1.2), shears=10.0, p=1.0),
        (2, 1, 5, 8, 9),
        # Translations (0.25 - 0.5) * 2 * 0.1 of the 9 x 8 x 5 volume, scale 0.8 + 0.25 * 0.4, angles -30 + 0.25 * 60
        # and shears -10 + 0.25 * 20 degrees.
        {
            "translations": [-0.45, -0.4, -0.25],
            "scale": [0.9, 0.9, 0.9],
            "angles": [-15.0, -15.0, -15.0],
            "sxy": -5.0,
            "sxz": -5.0,
            "syx": -5.0,
            "syz": -5.0,
            "szx": -5.0,
            "szy": -5.0,
        },
        id="affine3d",
    ),
    pytest.param(
        lambda: K.RandomRotation3D(30.0, p=1.0),
        (2, 1, 5, 8, 9),
        # -30 + 0.25 * 60 degrees about each axis.
        {"yaw": -15.0, "pitch": -15.0, "roll": -15.0},
        id="rotation3d",
    ),
    pytest.param(
        lambda: K.RandomMotionBlur3D(3, 35.0, 0.5, p=1.0),
        (2, 1, 5, 8, 9),
        # Angles -35 + 0.25 * 70 degrees and direction -0.5 + 0.25 * 1.
        {"angle_factor": [-17.5, -17.5, -17.5], "direction_factor": -0.25},
        id="motion-blur3d",
    ),
    pytest.param(
        lambda: K.RandomMotionBlur(3, 35.0, 0.5, p=1.0),
        (2, 3, 8, 9),
        # Angle -35 + 0.25 * 70 degrees and direction -0.5 + 0.25 * 1.
        {"angle_factor": -17.5, "direction_factor": -0.25},
        id="motion-blur",
    ),
    pytest.param(
        lambda: K.RandomSaltAndPepperNoise(amount=(0.01, 0.06), salt_vs_pepper=(0.4, 0.6), p=1.0),
        (2, 3, 8, 9),
        # Amount 0.01 + 0.25 * 0.05 and salt ratio 0.4 + 0.25 * 0.2.
        {"amount_factor": 0.0225, "salt_and_pepper_factor": 0.45},
        id="salt-and-pepper",
    ),
    pytest.param(
        lambda: K.RandomJPEG((20.0, 80.0), p=1.0),
        (2, 3, 8, 9),
        # 20 + 0.25 * 60.
        {"jpeg_quality": 35.0},
        id="jpeg",
    ),
    pytest.param(
        lambda: K.RandomRotation90((-3, 3), p=1.0),
        (2, 3, 8, 9),
        # The draw covers [-3, 4): floor(-3 + 0.25 * 7) = -2 quarter turns.
        {"times": -2.0},
        id="rotation90",
    ),
    pytest.param(
        lambda: K.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), p=1.0),
        (2, 3, 8, 9),
        # Area (0.02 + 0.25 * 0.31) * 72 = 7.02. The index draw rounds to 0, so the ratio comes from [1, 3.3):
        # 1 + 0.25 * 2.3 = 1.575. That gives a 2 x 3 rectangle (rounded), starting at floor(0.25 * 8) = 2 in x and
        # floor(0.25 * 6) = 1 in y.
        {"widths": 2.0, "heights": 3.0, "xs": 2.0, "ys": 1.0},
        id="erasing",
    ),
]


@pytest.mark.parametrize("make_aug, shape, expected", CASES)
class TestSampledParameterTransfer(BaseTester):
    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
    def test_sampled_values_survive_the_copy(self, make_aug, shape, expected, device, default_dtype, monkeypatch):
        # The sampler draws on the device in float32 and the parameters come back on the CPU in the default dtype.
        # With a float64 default on MPS each sampled value crosses from MPS to a float64 CPU tensor: torch 2.14 left
        # it unwritten (zeros here) and torch 2.15 raises (pytorch/pytorch#197715).
        aug = make_aug().to(device=device, dtype=torch.float32)
        real_rand = torch.rand
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.full_like(real_rand(*args, **kwargs), 0.25))
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(default_dtype)
            params = aug.forward_parameters(shape)
        finally:
            torch.set_default_dtype(original_dtype)
        for name, value in expected.items():
            actual = params[name]
            assert actual.device == torch.device("cpu"), name
            assert actual.dtype == default_dtype, name
            self.assert_close(actual, torch.tensor(value, dtype=default_dtype).expand_as(actual), rtol=0, atol=1e-6)
