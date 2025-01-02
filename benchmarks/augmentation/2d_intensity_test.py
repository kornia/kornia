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

from kornia.augmentation import (
    ColorJiggle,
    ColorJitter,
    Denormalize,
    Normalize,
    RandomAutoContrast,
    RandomBoxBlur,
    RandomBrightness,
    RandomChannelDropout,
    RandomChannelShuffle,
    RandomClahe,
    RandomContrast,
    RandomEqualize,
    RandomGamma,
    RandomGaussianBlur,
    RandomGaussianIllumination,
    RandomGaussianNoise,
    RandomGrayscale,
    RandomHue,
    RandomInvert,
    RandomLinearCornerIllumination,
    RandomLinearIllumination,
    RandomMedianBlur,
    RandomMotionBlur,
    RandomPlanckianJitter,
    RandomPlasmaBrightness,
    RandomPlasmaContrast,
    RandomPlasmaShadow,
    RandomPosterize,
    RandomRain,
    RandomRGBShift,
    RandomSaltAndPepperNoise,
    RandomSaturation,
    RandomSharpness,
    RandomSnow,
    RandomSolarize,
)


def test_aug_2d_collor_jiggle(benchmark, device, dtype, torch_optimizer, shape):
    if shape[1] != 3:
        pytest.skip("Skipping because input should be rgb")
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_collor_jitter(benchmark, device, dtype, torch_optimizer, shape):
    if shape[1] != 3:
        pytest.skip("Skipping because input should be rgb")
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0)
    op = torch_optimizer(aug)
    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_denormalize(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = Denormalize(0.0, 1.0, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_auto_contrast(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomAutoContrast(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_box_blur(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomBoxBlur(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_brightness(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomBrightness((0.1, 1), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_channel_dropout(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomChannelDropout(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_channel_shuffle(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomChannelShuffle(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_clahe(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomClahe((10, 40), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_contrast(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomContrast((0.1, 1), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_equalize(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomEqualize(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_gamma(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomGamma((0.0, 1.0), (0.0, 1.0), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_gaussian_blur(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomGaussianBlur(3, (1.6, 1.7), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_gaussian_illumination(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomGaussianIllumination(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_gaussian_noise(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomGaussianNoise(0.0, 1.0, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_grayscale(benchmark, device, dtype, torch_optimizer, shape):
    if shape[1] != 3:
        pytest.skip("Skipping because input should be rgb")
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomGrayscale(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_hue(benchmark, device, dtype, torch_optimizer, shape):
    if shape[1] != 3:
        pytest.skip("Skipping because input should be rgb")
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomHue((0.0, 0.5), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_invert(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomInvert(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


# TODO(joao.amorim): figure out dynamo issue with it
# def test_aug_2d_jpeg(benchmark, device, dtype, torch_optimizer, shape):
#     data = torch.rand(*shape, device=device, dtype=dtype)
#     aug = RandomJPEG(p=1.0)
#     op = torch_optimizer(aug)
#
#     actual = benchmark(op, input=data)
#
#     assert actual.shape == shape


def test_aug_2d_linear_corner_illumination(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomLinearCornerIllumination(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_linear_illumination(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomLinearIllumination(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_median_blur(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomMedianBlur(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_motion_blur(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomMotionBlur((3, 3), 45.0, 5.5, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_normalize(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = Normalize(25.0, 2.5, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_plackian_jitter(benchmark, device, dtype, torch_optimizer, shape):
    if shape[1] != 3:
        pytest.skip("Skipping because input should be rgb")

    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomPlanckianJitter(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_plasma_briggtness(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomPlasmaBrightness(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_plasma_contrast(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomPlasmaContrast(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_plasma_shadow(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomPlasmaShadow(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_posterize(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomPosterize(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_rain(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomRain(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_rgb_shift(benchmark, device, dtype, torch_optimizer, shape):
    if shape[1] != 3:
        pytest.skip("Skipping because input should be rgb")
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomRGBShift(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_salt_and_peper_noise(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomSaltAndPepperNoise(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_saturation(benchmark, device, dtype, torch_optimizer, shape):
    if shape[1] != 3:
        pytest.skip("Skipping because input should be rgb")
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomSaturation((0.0, 2.0), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_sharpness(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomSharpness(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_snow(benchmark, device, dtype, torch_optimizer, shape):
    if shape[1] != 3:
        pytest.skip("Skipping because input should be rgb")

    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomSnow(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_solarize(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomSolarize(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape
