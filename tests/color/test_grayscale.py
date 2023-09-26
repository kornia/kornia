from kornia.color.gray import grayscale_from_rgb
from testing.base_test import BaseTester


class TestColorGray(BaseTester):
    def test_channels_first(self):
        image_rgb = self.generate_random_image_data(shape=(3, 4, 5))
        image_gray = grayscale_from_rgb(image_rgb, channels_axis=0)
        assert image_gray.shape == (1, 4, 5)

    def test_channels_last(self):
        image_rgb = self.generate_random_image_data(shape=(4, 5, 3))
        image_gray = grayscale_from_rgb(image_rgb, channels_axis=2)
        assert image_gray.shape == (4, 5, 1)
