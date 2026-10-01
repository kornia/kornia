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

from kornia.image import image_to_string, print_image


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

    @pytest.mark.parametrize("max_width", [256, 3])
    @pytest.mark.parametrize("shape", [(1, 5, 6), (1, 4, 4), (1, 1, 2), (1, 7, 1)])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.uint8, torch.int64])
    def test_grayscale_matches_its_rgb_copy(self, shape, input_dtype, max_width):
        generator = torch.Generator().manual_seed(0)
        if input_dtype.is_floating_point:
            gray = torch.rand(shape, generator=generator, dtype=input_dtype)
        else:
            gray = torch.randint(0, 256, shape, generator=generator).to(input_dtype)
        out = image_to_string(gray, max_width)
        assert out  # an empty string would make the comparison below vacuous
        assert out == image_to_string(gray.repeat(3, 1, 1), max_width)

    def test_exception(self):
        img = torch.rand(3, 15, 15)
        image_to_string(img)

        img = torch.rand(3, 15, 15)
        image_to_string(img, max_width=12)

        from kornia.core.exceptions import ShapeError

        img = torch.rand(1, 3, 15, 15)
        with pytest.raises(ShapeError) as errinfo:
            image_to_string(img)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        from kornia.core.exceptions import ValueCheckError

        img = torch.rand(3, 15, 15) * 10
        with pytest.raises(ValueCheckError) as errinfo:
            image_to_string(img)
        assert "Value range mismatch" in str(errinfo.value) or "Invalid image value range" in str(errinfo.value)

        with pytest.raises(RuntimeError):
            print_image([img])  # Do not accept list

    def test_print_smoke(self):
        img = torch.rand(3, 15, 15)
        print_image(img)
