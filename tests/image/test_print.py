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

from kornia.image import image_print, image_to_string, print_image
from kornia.io import write_image


class TestImageToString:
    @pytest.mark.parametrize(("height", "rows"), [(2, 1), (3, 1), (6, 1), (10, 2)])
    def test_wide_image_keeps_at_least_one_row(self, height, rows, device, dtype):
        # the resized height is max(1, H * max_width // W): a wide, short image keeps one row, taller ones are unchanged
        image = torch.rand(3, height, 40, device=device, dtype=dtype)
        out = image_to_string(image, max_width=10)
        assert out.count("\n") == rows
        assert out.count("\033[48;5;") == rows * 10

    def test_value(self):
        image = torch.arange(16).reshape(1, 4, 4).repeat(3, 1, 1).long() * 16
        out = image_to_string(image)

        # nearest of the 240 fixed colours for the grays 0, 16, ..., 240 (code 59 = #5f5f5f and 145 = #afafaf are
        # cube colours that are nearer than any ramp gray); generated with
        #   from kornia.image.image_print import SHORT2RGB_DICT
        #   rgb = {c: [int(SHORT2RGB_DICT[str(c)][i : i + 2], 16) for i in (0, 2, 4)] for c in range(16, 256)}
        #   [min(rgb, key=lambda c: sum((a - 16 * i) ** 2 for a in rgb[c])) for i in range(16)]
        # (min takes the lowest code on a tie, as image_to_string does)
        expected = (
            "\033[48;5;16m  \033[48;5;233m  \033[48;5;234m  \033[48;5;236m  \033[0m\n"
            "\033[48;5;238m  \033[48;5;239m  \033[48;5;59m  \033[48;5;242m  \033[0m\n"
            "\033[48;5;244m  \033[48;5;246m  \033[48;5;247m  \033[48;5;145m  \033[0m\n"
            "\033[48;5;250m  \033[48;5;252m  \033[48;5;254m  \033[48;5;255m  \033[0m\n"
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

        with pytest.raises(TypeError):
            print_image([img])  # Do not accept list

    def test_rejects_a_tensor_that_is_not_a_color_or_gray_image(self):
        from kornia.core.exceptions import ImageError

        with pytest.raises(ImageError):
            image_to_string(torch.rand(6, 4, 4))

    def test_print_smoke(self):
        img = torch.rand(3, 15, 15)
        print_image(img)


class TestNearestXterm:
    # Expected codes generated with the snippet in TestImageToString.test_value, for the gray v:
    #   min(rgb, key=lambda c: sum((a - v) ** 2 for a in rgb[c]))  # v = 128 -> 244, v = 48 -> 236

    def test_gray_uses_the_gray_ramp(self):
        # 128 -> code 244 (#808080, exact); the 6-level cube's nearest was code 102 (#878787)
        out = image_to_string(torch.full((3, 2, 3), 128, dtype=torch.uint8))
        assert out.count("48;5;244m") == 6
        assert out.count("\n") == 2

    def test_levels_round_instead_of_truncating(self):
        # float 0.1875 = 47.8 levels -> 48 -> nearest of codes 16-255 is code 236 (#303030, exact); truncation gave
        # code 16
        out = image_to_string(torch.full((3, 2, 3), 0.1875))
        assert out.count("48;5;236m") == 6
        assert "48;5;16m" not in out

    def test_tall_image_equals_its_two_halves(self):
        # the nearest colours are computed in chunks of pixels; an image taller than one chunk renders exactly like its
        # halves rendered one after the other, so no pixel is lost or moved at a chunk boundary (mid-row here)
        width = 250  # below max_width, so nothing is resized
        height = 2 * (image_print._PIXELS_PER_CHUNK // width + 1)
        gen = torch.Generator().manual_seed(0)
        image = torch.randint(0, 256, (3, height, width), generator=gen, dtype=torch.uint8)
        top, bottom = image[:, : height // 2], image[:, height // 2 :]
        assert image_to_string(image) == image_to_string(top) + image_to_string(bottom)

    def test_an_image_on_the_device_prints_like_its_cpu_copy(self, device):
        # the distances run on the CPU in float64; MPS cannot hold float64, and a combined .to("cpu", float64) gives
        # zeros there on torch 2.14 (every pixel black) and raises TypeError on torch 2.5.1
        image = torch.randint(0, 256, (3, 4, 5), generator=torch.Generator().manual_seed(0), dtype=torch.uint8)
        out = image_to_string(image.to(device))
        assert out == image_to_string(image)
        assert out.count("48;5;16m") < 20  # not every pixel black

    def test_distances_are_computed_in_bounded_chunks(self, monkeypatch):
        # a 1500x256 image used to build one (N, 240) float64 distance matrix (about 1.7 GB peak)
        sizes = []
        nearest = image_print._nearest_xterm_codes

        def record(rgb):
            sizes.append(rgb.shape[0])
            return nearest(rgb)

        monkeypatch.setattr(image_print, "_nearest_xterm_codes", record)
        image = torch.zeros(3, 2 * (image_print._PIXELS_PER_CHUNK // 250 + 1), 250, dtype=torch.uint8)
        image_to_string(image)
        assert max(sizes) <= image_print._PIXELS_PER_CHUNK
        assert sum(sizes) == image[0].numel()

    def test_print_image_accepts_a_path(self, tmp_path, capsys):
        write_image(tmp_path / "p.png", torch.zeros(3, 2, 3, dtype=torch.uint8))
        print_image(tmp_path / "p.png")
        assert "\x1b[48;5;" in capsys.readouterr().out
