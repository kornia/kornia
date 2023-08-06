import os

import jax
import numpy as np
import pytest
import torch

from kornia.color.gray_core import grayscale_to_rgb
from kornia.core import Device, Dtype
from kornia.image.base import ChannelsOrder, ColorSpace, ImageLayout, ImageSize, PixelFormat
from kornia.image.image import Image
from kornia.testing import BaseTester

# TODO: this should be done in the conftest.py
os.environ["KERAS_BACKEND"] = "torch"


def _make_random_image_u8(shape, device, dtype, backend):
    if backend == "torch":
        return torch.randint(0, 255, shape, device=device)
    elif backend == "jax":
        return jax.random.randint(jax.random.PRNGKey(0), shape=shape, minval=0, maxval=255)
    elif backend == "numpy":
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, shape, dtype=np.uint8)
    else:
        raise NotImplementedError(backend)


class TestGrayCore(BaseTester):
    # @pytest.mark.parametrize("backend", ["torch"])
    # @pytest.mark.parametrize("backend", ["numpy"])
    @pytest.mark.parametrize("backend", ["jax"])
    @pytest.mark.parametrize("shape", [(4, 5), (5, 4)])
    @pytest.mark.parametrize("num_channels", [1])
    def test_smoke(self, backend, device, dtype, num_channels, shape):
        # NOTE: just to make sure that the function is working
        if backend in ["torch", "jax"]:
            _shape = (num_channels, *shape)
            channels_order = ChannelsOrder.CHANNELS_FIRST
        else:
            _shape = (*shape, num_channels)
            channels_order = ChannelsOrder.CHANNELS_LAST
        image_gray_data = _make_random_image_u8(_shape, device, dtype, backend)
        image_gray = Image(
            data=image_gray_data,
            pixel_format=PixelFormat(color_space=ColorSpace.GRAY, bit_depth=8),
            layout=ImageLayout(
                image_size=ImageSize(height=shape[0], width=shape[1]),
                channels=num_channels,
                channels_order=channels_order,
            ),
        )

        image_rgb: Image = grayscale_to_rgb(image_gray)
        assert image_rgb.layout.image_size == image_gray.layout.image_size
        assert image_rgb.pixel_format.color_space == ColorSpace.RGB
        assert image_rgb.layout.channels == 3

        # TODO: make this generic with assert_allclose
        # for i in range(3):
        #     assert image_rgb.get_channel(i) == image_gray.get_channel(0)

    def test_cardinality(self, device: Device, dtype: Dtype) -> None:
        pass

    def test_exception(self, device: Device, dtype: Dtype) -> None:
        pass

    def test_gradcheck(self, device: Device) -> None:
        pass

    def test_module(self, device: Device, dtype: Dtype) -> None:
        pass
