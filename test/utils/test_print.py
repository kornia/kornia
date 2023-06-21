import pytest
import torch

from kornia.utils import image_to_string


class TestImageToString:

    def test_image_to_string(self):
        img = torch.rand(3, 15, 15)
        image_to_string(img)

        with pytest.raises(AssertionError):
            img = torch.rand(1, 3, 15, 15)
            image_to_string(img)

        with pytest.raises(ValueError):
            img = torch.rand(3, 15, 15) * 10
            image_to_string(img)
