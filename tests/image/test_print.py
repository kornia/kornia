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

import os
import re

import pytest
import torch

from kornia.core.exceptions import ValueCheckError
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
    # Expected codes generated with the snippet in TestImageToString.test_value, for the 0-255 pixel value p:
    #   min(rgb, key=lambda c: sum((a - b) ** 2 for a, b in zip(rgb[c], p)))
    # p = (128,) * 3 -> 244, (91.6,) * 3 -> 59, (91,) * 3 -> 240, (4.49, 4.49, 3.1) -> 232, (4, 4, 3) -> 16

    def test_gray_uses_the_gray_ramp(self):
        # 128 -> code 244 (#808080, exact) in the gray ramp; the nearest cube colour is code 102 (#878787)
        out = image_to_string(torch.full((3, 2, 3), 128, dtype=torch.uint8))
        assert out.count("48;5;244m") == 6
        assert out.count("\n") == 2

    def test_a_float_level_is_not_truncated(self):
        # 91.6 levels: the nearest code is 59 (#5f5f5f = 95, 3.4 away), not 240 (#585858 = 88, 3.6 away), which a
        # level truncated to 91 would pick
        out = image_to_string(torch.full((3, 2, 3), 91.6 / 255))
        assert out.count("48;5;59m") == 6
        assert "48;5;240m" not in out

    def test_a_float_pixel_is_compared_at_its_unrounded_value(self):
        # (4.49, 4.49, 3.1) levels: code 232 (#080808) is nearer than code 16 (#000000), squared distance 48.65
        # against 49.93; the pixel rounded to (4, 4, 3) would pick code 16
        image = (torch.tensor([4.49, 4.49, 3.1]) / 255)[:, None, None].expand(3, 2, 3)
        out = image_to_string(image)
        assert out.count("48;5;232m") == 6

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
        # the distances run on the CPU in float64, which MPS cannot hold: the pixels move to the CPU before the cast
        image = torch.randint(0, 256, (3, 4, 5), generator=torch.Generator().manual_seed(0), dtype=torch.uint8)
        out = image_to_string(image.to(device))
        assert out == image_to_string(image)
        assert out.count("48;5;16m") < 20  # not every pixel black

    def test_distances_are_computed_in_bounded_chunks(self, monkeypatch):
        # no (pixels, 240) distance matrix has more rows than _PIXELS_PER_CHUNK, whatever the image size
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

    def test_print_image_accepts_a_path_like_returning_bytes(self, tmp_path, capsys):
        path = tmp_path / "p.png"
        write_image(path, torch.zeros(3, 2, 3, dtype=torch.uint8))

        class BytesPath(os.PathLike):
            def __fspath__(self):
                return os.fsencode(path)

        print_image(BytesPath())
        out = capsys.readouterr().out
        assert out.count("\x1b[48;5;16m") == 6  # 2 x 3 black pixels
        assert out.count("\x1b[0m\n") == 2


class TestConventionsPrint:
    """Pins for the value ranges and the width defaults of the terminal printers."""

    @pytest.mark.parametrize(
        ("image", "expected_range"),
        [
            (torch.full((3, 2, 4), 1.5), "expected [0, 1]"),
            (torch.full((3, 2, 4), -0.5), "expected [0, 1]"),
            (torch.full((3, 2, 4), 256, dtype=torch.int64), "expected [0, 255]"),
        ],
        ids=["float-above-1", "float-below-0", "int-above-255"],
    )
    def test_convention_image_to_string_value_ranges(self, image, expected_range):
        # A float image is read as [0, 1] and an integer one as [0, 255]; anything outside raises.
        with pytest.raises(ValueCheckError, match=re.escape(expected_range)):
            image_to_string(image)

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int64])
    def test_convention_image_to_string_integer_255_is_float_1(self, dtype):
        # The accepted boundaries: integer 255 is the same colour as float 1.0, so integers are not read as [0, 1].
        assert image_to_string(torch.full((3, 2, 4), 255, dtype=dtype)) == image_to_string(torch.ones(3, 2, 4))

    def test_convention_print_max_width_defaults(self, capsys):
        # image_to_string caps the width at 256 pixels by default and print_image at 96. Each printed pixel is one
        # "\x1b[48;5;<code>m  " escape, so a 300-pixel-wide image prints 256 or 96 cells per row.
        image = torch.zeros(3, 2, 300)
        string = image_to_string(image)
        assert string.count("\033[0m\n") == 1  # H = 2 shrinks to max(1, 2 * 256 // 300) = 1 row
        assert string.count("\033[48;5;") == 256
        print_image(image)
        printed = capsys.readouterr().out
        assert printed.count("\033[0m\n") == 1  # max(1, 2 * 96 // 300) = 1 row
        assert printed.count("\033[48;5;") == 96
