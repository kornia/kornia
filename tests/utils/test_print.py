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

    def test_exception(self):
        img = torch.rand(3, 15, 15)
        image_to_string(img)

        img = torch.rand(3, 15, 15)
        image_to_string(img, max_width=12)

        from kornia.core.exceptions import ShapeError

        with pytest.raises(ShapeError) as errinfo:
            img = torch.rand(1, 3, 15, 15)
            image_to_string(img)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        from kornia.core.exceptions import ValueCheckError

        with pytest.raises(ValueCheckError) as errinfo:
            img = torch.rand(3, 15, 15) * 10
            image_to_string(img)
        assert "Value range mismatch" in str(errinfo.value) or "Invalid image value range" in str(errinfo.value)

        with pytest.raises(RuntimeError):
            print_image([img])  # Do not accept list

    def test_print_smoke(self):
        img = torch.rand(3, 15, 15)
        print_image(img)
