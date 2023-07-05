import pytest
import torch

from kornia.utils import image_to_string, print_image


class TestImageToString:
    def test_value(self):
        image = torch.arange(16).reshape(1, 4, 4).repeat(3, 1, 1).long() * 16
        out = image_to_string(image)

        expected = (
            "\033[48;5;16m  \033[48;5;16m  \033[48;5;16m  \033[48;5;59m  \033[0m\n"
            "\033[48;5;59m  \033[48;5;59m  \033[48;5;59m  \033[48;5;59m  \033[0m\n"
            "\033[48;5;102m  \033[48;5;102m  \033[48;5;145m  \033[48;5;145m  \033[0m\n"
            "\033[48;5;145m  \033[48;5;188m  \033[48;5;188m  \033[48;5;231m  \033[0m\n"
        )
        assert out == expected

    def test_errors(self):
        img = torch.rand(3, 15, 15)
        image_to_string(img)

        img = torch.rand(3, 15, 15)
        image_to_string(img, max_width=12)

        with pytest.raises(TypeError):
            img = torch.rand(1, 3, 15, 15)
            image_to_string(img)

        with pytest.raises(ValueError):
            img = torch.rand(3, 15, 15) * 10
            image_to_string(img)

    def test_print_smoke(self):
        img = torch.rand(3, 15, 15)
        print_image(img)

        with pytest.raises(RuntimeError):
            print_image([img])  # Do not accept list
