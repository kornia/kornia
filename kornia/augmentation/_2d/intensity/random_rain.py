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

from __future__ import annotations

from typing import Any, Optional

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.random_generator._2d import RainGenerator
from kornia.core.check import KORNIA_CHECK


class RandomRain(IntensityAugmentationBase2D):
    r"""Add Random Rain to the image.

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        p: probability of applying the transformation.
        number_of_drops: number of drops per image
        drop_height: Height of the drop in the image (same for each drop in one image). Sampled values must be
            greater than zero and strictly smaller than the input image height.
        drop_width: Width of the drop in the image (same for each drop in one image). The absolute sampled value
            must be strictly smaller than the input image width.
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Convention:
        - the input must have one or three channels; any other channel count raises on the forward pass.
        - ``drop_height`` runs down rows and ``drop_width`` along columns: both name image axes, not
          drop-local ones, and a negative ``drop_width`` slants the drop the other way across the columns.
        - a drop is written as the fixed value ``200 / 255``, not as a function of the image, so the rain is
          darker than every pixel above ``200 / 255`` that it falls on, inside ``[0, 1]`` or not. Every other
          pixel is carried through unclamped.
        - both sizes must be strictly smaller than the image on their own axis, checked on the forward pass,
          where the image shape is known; a ``drop_height`` below ``1`` raises there too.
        - every start position that keeps the whole drop inside the image is equally likely.
        - the three integer ranges are closed and uniform, so the default ``drop_height=(5, 20)`` reaches ``20``
          and the default ``drop_width=(-5, 5)`` reaches both ends. With the defaults an image 20 pixels tall, or
          5 pixels wide, therefore raises on some draws. A range that is reversed, fractional or non-finite
          raises ``ValueError`` at construction.

    .. warning::
        A drop whose larger size ``n`` is at least ``2`` paints ``n`` pixels spread over ``n + 1`` rows or
        columns, so it has a one-pixel gap: ``drop_height=5`` with ``drop_width=0`` paints rows
        ``[0, 1, 2, 3, 5]``. Tracked in `#4810 <https://github.com/kornia/kornia/issues/4810>`_.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> rain = RandomRain(p=1,drop_height=(1,2),drop_width=(1,2),number_of_drops=(1,1))
        >>> rain(input)
        tensor([[[[0.4963, 0.7843, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.7843, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])

    """

    def __init__(
        self,
        number_of_drops: tuple[int, int] = (1000, 2000),
        drop_height: tuple[int, int] = (5, 20),
        drop_width: tuple[int, int] = (-5, 5),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = RainGenerator(number_of_drops, drop_height, drop_width)

    def apply_transform(
        self,
        image: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Check array and drops size
        KORNIA_CHECK(image.shape[1] in {3, 1}, "Number of color channels should be 1 or 3.")
        KORNIA_CHECK(
            bool(
                torch.all(params["drop_height_factor"] < image.shape[2]) and torch.all(params["drop_height_factor"] > 0)
            ),
            "Height of drop should be greater than zero and less than image height.",
        )

        KORNIA_CHECK(
            bool(torch.all(torch.abs(params["drop_width_factor"]) < image.shape[3])),
            "Width of drop should be less than image width.",
        )
        output = image.clone()
        batch_size, _, image_height, image_width = image.shape
        coordinates: torch.Tensor = params["coordinates_factor"]  # (B, max drops, 2) in [0, 1]
        max_drops = coordinates.shape[1]
        if batch_size == 0 or max_drops == 0:
            return output

        # The three per-sample integers are read once for the whole batch: one device sync per
        # parameter instead of three per sample. `int()` truncates exactly as `int(params[...][i])`
        # did per element.
        drops_per_sample = [int(v) for v in params["number_of_drops_factor"].tolist()]
        heights = [int(v) for v in params["drop_height_factor"].tolist()]
        widths = [int(v) for v in params["drop_width_factor"].tolist()]

        # Every drop of every sample is rasterised in one batched computation and written with ONE
        # indexed assignment (#4530). Before, each sample paid `size_of_line` separate `index_put_`
        # launches, one per step of the line, plus its own host-to-device copies: ~100 writes and
        # 204 adds at batch 8. The per-sample work that remains below runs on the host and touches
        # no device: it builds the line shape for each sample exactly as before, with the same
        # `torch.linspace` in the same dtype, so the pixels are the ones the step loop painted, and
        # all are set to one constant, so write order cannot change the result.
        #
        # The line shape differs per sample (its height, width and step count are all drawn), so the
        # shapes are padded to the longest line in the batch and a mask drops the padding. The staging
        # buffers are pinned to the CPU explicitly rather than left to the default device, so a
        # `torch.set_default_device` in the caller cannot turn the host loop into per-sample device
        # launches, or a copy from `meta` that cannot be made.
        longest_line = max(max(h, abs(w)) for h, w in zip(heights, widths))
        lines = torch.zeros(batch_size, 2, longest_line, dtype=torch.long, device="cpu")
        # One row per sample, packed so the host-side integers cross to the device in a single copy:
        # [admissible start rows, admissible start cols, col shift, line length, drop count].
        # The admissible start region: a drop may start anywhere its far end stays inside the image.
        # The far end is the line's last offset -- the end point is included once there are at least
        # two steps, and a single-pixel drop is the start alone. Derived from the Python ints rather
        # than read off `x[-1]`/`y[-1]`, which would sync the device per sample.
        meta = torch.empty(batch_size, 5, dtype=torch.long, device="cpu")
        for i, (height_of_drop, width_of_drop) in enumerate(zip(heights, widths)):
            # Generate how our drop will look like into the image
            size_of_line = max(height_of_drop, abs(width_of_drop))
            lines[i, 0, :size_of_line] = torch.linspace(
                0, height_of_drop, steps=size_of_line, dtype=torch.long, device="cpu"
            )
            lines[i, 1, :size_of_line] = torch.linspace(
                0, width_of_drop, steps=size_of_line, dtype=torch.long, device="cpu"
            )
            last_dy, last_dx = (height_of_drop, width_of_drop) if size_of_line > 1 else (0, 0)
            meta[i, 0] = image_height - last_dy
            meta[i, 1] = image_width - abs(last_dx)
            meta[i, 2] = max(-last_dx, 0)
            meta[i, 3] = size_of_line
            meta[i, 4] = drops_per_sample[i]

        # Start coordinates, computed on the device the draw lives on so the float product rounds as
        # it always has, then moved once. The clamp is for the MPS half `rand` that can return exactly
        # 1.0 (#4553): it keeps such a draw on the last admissible start instead of one past it.
        meta_c = meta.to(coordinates.device)
        rows_c, cols_c, shift_c = meta_c[:, 0:1], meta_c[:, 1:2], meta_c[:, 2:3]
        start_rows = torch.minimum((coordinates[..., 0] * rows_c).long(), rows_c - 1)
        start_cols = torch.minimum((coordinates[..., 1] * cols_c).long(), cols_c - 1) + shift_c

        # Two device copies for the whole batch: the host-built shapes and metadata as one flat
        # buffer, and the two start-coordinate grids as one stacked tensor.
        device = image.device
        host = torch.cat([lines.reshape(-1), meta.reshape(-1)]).to(device)
        lines = host[: lines.numel()].view(batch_size, 2, longest_line)
        meta = host[lines.numel() :].view(batch_size, 5)
        starts = torch.stack([start_rows, start_cols]).to(device)
        # (B, max drops, longest line): every start offset by every step of its sample's line.
        drop_rows = starts[0].unsqueeze(2) + lines[:, 0].unsqueeze(1)
        drop_cols = starts[1].unsqueeze(2) + lines[:, 1].unsqueeze(1)

        # Keep drop j of sample i only while j < its drawn drop count (the generator allocates the
        # batch maximum), and step k only while k < its line length (padding above). The mask is
        # resolved with a single `nonzero` (one device sync) and the batch index is recovered from
        # the flat position arithmetically, rather than boolean-indexing three tensors, which would
        # call `nonzero` three times.
        drop_valid = torch.arange(max_drops, device=device).unsqueeze(0) < meta[:, 4:5]
        step_valid = torch.arange(longest_line, device=device).unsqueeze(0) < meta[:, 3:4]
        valid = drop_valid.unsqueeze(2) & step_valid.unsqueeze(1)
        flat = valid.reshape(-1).nonzero().squeeze(1)
        batch_index = torch.div(flat, max_drops * longest_line, rounding_mode="floor")

        output[batch_index, :, drop_rows.reshape(-1)[flat], drop_cols.reshape(-1)[flat]] = 200 / 255
        return output
