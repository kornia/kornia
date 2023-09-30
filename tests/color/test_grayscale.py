from kornia.color.gray import grayscale_from_rgb
from testing.base import BaseTester


class TestColorGray(BaseTester):
    def test_smoke(self, channels_axis, dtype):
        image_rgb = self.generate_random_image(None, 4, 5, dtype=dtype)
        image_gray = grayscale_from_rgb(image_rgb)
        expected_shape = self.load_shape(None, 4, 5, 1)
        assert image_gray.shape == expected_shape

    def test_smoke_using_axis(self, channels_axis, dtype):
        image_rgb = self.generate_random_image(None, 4, 5, dtype=dtype)
        image_gray = grayscale_from_rgb(image_rgb, channels_axis=channels_axis)
        expected_shape = self.load_shape(None, 4, 5, 1)
        assert image_gray.shape == expected_shape
