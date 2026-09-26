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

"""In this module several equalization methods are exposed: he, ahe, clahe."""

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F

from kornia.core.utils import _normalize_to_float32_or_float64
from kornia.image.utils import perform_keep_shape_image

from .adjust import _lookup_value_check
from .histogram import histogram


def _compute_tiles(
    imgs: torch.Tensor, grid_size: Tuple[int, int], even_tile_size: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute tiles on an image according to a grid size.

    Note that padding can be added to the image in order to crop properly the image.
    So, the grid_size (GH, GW) x tile_size (TH, TW) >= image_size (H, W)

    Args:
        imgs: batch of 2D images with shape (B, C, H, W) or (C, H, W).
        grid_size: number of tiles to be cropped in each direction (GH, GW)
        even_tile_size: Determine if the width and height of the tiles must be even.

    Returns:
        tensor with tiles (B, GH, GW, C, TH, TW). B = 1 in case of a single image is provided.
        tensor with the padded batch of 2D imageswith shape (B, C, H', W').

    """
    batch: torch.Tensor = imgs  # B x C x H x W

    # compute stride and kernel size
    h, w = batch.shape[-2:]
    kernel_vert: int = math.ceil(h / grid_size[0])
    kernel_horz: int = math.ceil(w / grid_size[1])

    if even_tile_size:
        kernel_vert += 1 if kernel_vert % 2 else 0
        kernel_horz += 1 if kernel_horz % 2 else 0

    # add padding (with that kernel size we could need some extra cols and rows...)
    pad_vert = kernel_vert * grid_size[0] - h
    pad_horz = kernel_horz * grid_size[1] - w

    # add the padding in the last coluns and rows
    # reflect padding needs the pad strictly below the axis it reflects, so >= is the bound: an
    # 8 x 8 image on an 8 x 8 grid pads by 8 on an axis of 8 and fails inside F.pad otherwise.
    if pad_vert >= batch.shape[-2] or pad_horz >= batch.shape[-1]:
        # An even tile has to cover more than one grid cell, so the axis must exceed the grid;
        # an odd one only has to exceed half of it.
        min_vert = grid_size[0] + 1 if even_tile_size else grid_size[0] // 2 + 1
        min_horz = grid_size[1] + 1 if even_tile_size else grid_size[1] // 2 + 1
        raise ValueError(
            "Cannot compute tiles on the image according to the given grid size. "
            f"Got image size ({h}, {w}) and grid size {tuple(grid_size)}, which needs "
            f"({pad_vert}, {pad_horz}) of reflect padding, more than the image has to reflect. "
            f"The smallest image this grid admits is ({min_vert}, {min_horz})."
        )

    if pad_vert > 0 or pad_horz > 0:
        batch = F.pad(batch, [0, pad_horz, 0, pad_vert], mode="reflect")  # B x C x H' x W'

    # compute tiles
    c: int = batch.shape[-3]
    tiles: torch.Tensor = (
        batch.unfold(1, c, c)  # unfold(dimension, size, step)
        .unfold(2, kernel_vert, kernel_vert)
        .unfold(3, kernel_horz, kernel_horz)
        .squeeze(1)
    ).contiguous()  # GH x GW x C x TH x TW

    if tiles.shape[-5] != grid_size[0]:
        raise AssertionError

    if tiles.shape[-4] != grid_size[1]:
        raise AssertionError

    return tiles, batch


def _compute_interpolation_tiles(padded_imgs: torch.Tensor, tile_size: Tuple[int, int]) -> torch.Tensor:
    r"""Compute interpolation tiles on a properly padded set of images.

    Note that images must be padded. So, the tile_size (TH, TW) * grid_size (GH, GW) = image_size (H, W)

    Args:
        padded_imgs: batch of 2D images with shape (B, C, H, W) already padded to extract tiles
                                    of size (TH, TW).
        tile_size: shape of the current tiles (TH, TW).

    Returns:
        tensor with the interpolation tiles (B, 2GH, 2GW, C, TH/2, TW/2).

    """
    if padded_imgs.dim() != 4:
        raise AssertionError("Images Tensor must be 4D.")

    if padded_imgs.shape[-2] % tile_size[0] != 0:
        raise AssertionError("Images are not correctly padded.")

    if padded_imgs.shape[-1] % tile_size[1] != 0:
        raise AssertionError("Images are not correctly padded.")

    # tiles to be interpolated are built by dividing in 4 each already existing
    interp_kernel_vert: int = tile_size[0] // 2
    interp_kernel_horz: int = tile_size[1] // 2

    c: int = padded_imgs.shape[-3]
    interp_tiles: torch.Tensor = (
        padded_imgs.unfold(1, c, c)
        .unfold(2, interp_kernel_vert, interp_kernel_vert)
        .unfold(3, interp_kernel_horz, interp_kernel_horz)
        .squeeze(1)
    ).contiguous()  # 2GH x 2GW x C x TH/2 x TW/2

    if interp_tiles.shape[-3] != c:
        raise AssertionError

    if interp_tiles.shape[-2] != tile_size[0] / 2:
        raise AssertionError

    if interp_tiles.shape[-1] != tile_size[1] / 2:
        raise AssertionError

    return interp_tiles


def _tiles_histc(tiles: torch.Tensor, bins: int) -> torch.Tensor:
    r"""Histogram every tile over ``[0, 1]`` in one pass, matching a per-tile CPU ``torch.histc``.

    One ``scatter_add_`` counts all tiles, instead of one ``histc`` launch per tile, and keeps a static
    output shape so the function stays a single dynamo graph. The index is computed as ``histc``
    computes it, in the same float32/float64 promotion as ``_torch_histc_cast``: ``floor(x * bins)``,
    with ``x == 1`` in the last bin. Values outside ``[0, 1]`` and NaN are not counted, which the CPU
    ``histc`` also skips; they go to one extra overflow column per tile that is dropped. (``histc`` on
    MPS counts values marginally outside ``[0, 1]``, so there this matches CPU rather than MPS.)

    Args:
        tiles: flattened tiles. (T, P)
        bins: number of bins.

    Returns:
        Per-tile counts in the dtype of ``tiles``. (T, bins)

    """
    num_tiles = tiles.shape[0]
    x = tiles.to(_normalize_to_float32_or_float64(tiles.dtype))
    # clamp keeps [0, 1] unchanged and NaN as NaN, so the comparison is False exactly where histc skips.
    in_range = x.clamp(0, 1) == x
    idx = (x * bins).to(torch.int64).clamp_(0, bins - 1)
    # this mask is what drops out-of-range values; without it they land in the first or last bin
    idx = torch.where(in_range, idx, bins)
    counts = torch.zeros(num_tiles, bins + 1, dtype=x.dtype, device=x.device)
    counts.scatter_add_(1, idx, torch.ones_like(x))
    return counts[:, :bins].to(tiles.dtype)


def _compute_luts(
    tiles_x_im: torch.Tensor, num_bins: int = 256, clip: Union[float, torch.Tensor] = 40.0, diff: bool = False
) -> torch.Tensor:
    r"""Compute luts for a batched set of tiles.

    Same approach as in OpenCV (https://github.com/opencv/opencv/blob/master/modules/imgproc/src/clahe.cpp)

    Args:
        tiles_x_im: set of tiles per image to apply the lut. (B, GH, GW, C, TH, TW)
        num_bins: number of bins. default: 256
        clip: threshold value for contrast limiting. If it is 0 then the clipping is disabled.
        diff: denote if the differentiable histagram will be used. Default: False

    Returns:
        Lut for each tile (B, GH, GW, C, 256).

    """
    if tiles_x_im.dim() != 6:
        raise AssertionError("Tensor must be 6D.")

    b, gh, gw, c, th, tw = tiles_x_im.shape
    pixels: int = th * tw
    tiles: torch.Tensor = tiles_x_im.view(-1, pixels)  # test with view  # T x (THxTW)
    if not diff:
        histos = _tiles_histc(tiles, num_bins)
    else:
        bins: torch.Tensor = torch.linspace(0, 1, num_bins, device=tiles.device)
        # histogram already returns (T, num_bins); squeeze() dropped the tile axis for a single tile
        histos = histogram(tiles, bins, torch.tensor(0.001))
        histos *= pixels

    if isinstance(clip, torch.Tensor):
        # Match Python scalar arithmetic before rounding each image's threshold, then broadcast
        # over its tiles and channels. Keeping the limits in tensors avoids compile guards on draws.
        # MPS cannot store doubles; compute its thresholds on CPU before copying them with the LUTs.
        limits = clip.to(device="cpu", dtype=torch.float64) if clip.device.type == "mps" else clip.double()
        max_vals = (limits * pixels).div(num_bins, rounding_mode="floor").clamp(min=1)
        max_vals = max_vals.to(histos).view(b, 1).expand(b, gh * gw * c).reshape(-1, 1)
        limited = histos.clamp(max=max_vals)
        clipped_tensor = torch.relu(histos - max_vals).sum(1) if diff else pixels - limited.sum(1)
        residual_tensor = torch.remainder(clipped_tensor, num_bins)
        limited = limited + ((clipped_tensor - residual_tensor) / num_bins).unsqueeze(1)
        limited = limited + (torch.arange(num_bins, device=histos.device) < residual_tensor.unsqueeze(1))
        enabled = (clip > 0).to(device=histos.device).view(b, 1).expand(b, gh * gw * c).reshape(-1, 1)
        histos = torch.where(enabled, limited, histos)
    elif clip > 0.0:
        max_val: float = max(clip * pixels // num_bins, 1)
        if diff:
            clipped: torch.Tensor = torch.relu(histos - max_val).sum(1)
        histos.clamp_(max=max_val)
        if not diff:
            clipped = pixels - histos.sum(1)
        residual: torch.Tensor = torch.remainder(clipped, num_bins)
        redist: torch.Tensor = (clipped - residual).div(num_bins)
        histos += redist[None].transpose(0, 1)
        # trick to avoid using a loop to assign the residual
        v_range: torch.Tensor = torch.arange(num_bins, device=histos.device)
        mat_range: torch.Tensor = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)

    lut_scale: float = (num_bins - 1) / pixels
    luts: torch.Tensor = torch.cumsum(histos, 1) * lut_scale
    luts = luts.clamp(0, num_bins - 1)
    if not diff:
        luts = luts.floor()  # to get the same values as converting to int maintaining the type
    return luts.view((b, gh, gw, c, num_bins))


def _map_luts(interp_tiles: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    r"""Assign the required luts to each tile.

    Args:
        interp_tiles: set of interpolation tiles. (B, 2GH, 2GW, C, TH/2, TW/2)
        luts: luts for each one of the original tiles. (B, GH, GW, C, 256)

    Returns:
         mapped luts (B, 2GH, 2GW, 4, C, 256)

    """
    if interp_tiles.dim() != 6:
        raise AssertionError("interp_tiles tensor must be 6D.")

    if luts.dim() != 5:
        raise AssertionError("luts tensor must be 5D.")

    # gh, gw -> 2x the number of tiles used to compute the histograms
    # th, tw -> /2 the sizes of the tiles used to compute the histograms
    num_imgs, gh, gw, c, _, _ = interp_tiles.shape

    # precompute idxs for non corner regions (doing it in cpu seems slightly faster)
    j_idxs = torch.empty(0, 4, dtype=torch.long)
    if gh > 2:
        j_floor = torch.arange(1, gh - 1).view(gh - 2, 1).div(2, rounding_mode="trunc")
        j_idxs = torch.tensor([[0, 0, 1, 1], [-1, -1, 0, 0]] * ((gh - 2) // 2))  # reminder + j_idxs[:, 0:2] -= 1
        j_idxs += j_floor

    i_idxs = torch.empty(0, 4, dtype=torch.long)
    if gw > 2:
        i_floor = torch.arange(1, gw - 1).view(gw - 2, 1).div(2, rounding_mode="trunc")
        i_idxs = torch.tensor([[0, 1, 0, 1], [-1, 0, -1, 0]] * ((gw - 2) // 2))  # reminder + i_idxs[:, [0, 2]] -= 1
        i_idxs += i_floor

    # selection of luts to interpolate each patch
    # create a tensor with dims: interp_patches height and width x 4 x num channels x bins in the histograms
    # the tensor is init to -1 to denote non init hists
    luts_x_interp_tiles: torch.Tensor = torch.full(  # B x GH x GW x 4 x C x 256
        (num_imgs, gh, gw, 4, c, luts.shape[-1]), -1, dtype=interp_tiles.dtype, device=interp_tiles.device
    )
    # corner regions
    luts_x_interp_tiles[:, 0 :: gh - 1, 0 :: gw - 1, 0] = luts[:, 0 :: max(gh // 2 - 1, 1), 0 :: max(gw // 2 - 1, 1)]
    # border region (h)
    luts_x_interp_tiles[:, 1:-1, 0 :: gw - 1, 0] = luts[:, j_idxs[:, 0], 0 :: max(gw // 2 - 1, 1)]
    luts_x_interp_tiles[:, 1:-1, 0 :: gw - 1, 1] = luts[:, j_idxs[:, 2], 0 :: max(gw // 2 - 1, 1)]
    # border region (w)
    luts_x_interp_tiles[:, 0 :: gh - 1, 1:-1, 0] = luts[:, 0 :: max(gh // 2 - 1, 1), i_idxs[:, 0]]
    luts_x_interp_tiles[:, 0 :: gh - 1, 1:-1, 1] = luts[:, 0 :: max(gh // 2 - 1, 1), i_idxs[:, 1]]
    # internal region
    luts_x_interp_tiles[:, 1:-1, 1:-1, :] = luts[
        :, j_idxs.repeat(max(gw - 2, 1), 1, 1).permute(1, 0, 2), i_idxs.repeat(max(gh - 2, 1), 1, 1)
    ]

    return luts_x_interp_tiles


def _compute_equalized_tiles(interp_tiles: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    r"""Equalize the tiles.

    Args:
        interp_tiles: set of interpolation tiles, values must be in the range [0, 1].
          (B, 2GH, 2GW, C, TH/2, TW/2)
        luts: luts for each one of the original tiles. (B, GH, GW, C, 256)

    Returns:
        equalized tiles (B, 2GH, 2GW, C, TH/2, TW/2)

    """
    if interp_tiles.dim() != 6:
        raise AssertionError("interp_tiles tensor must be 6D.")

    if luts.dim() != 5:
        raise AssertionError("luts tensor must be 5D.")

    mapped_luts: torch.Tensor = _map_luts(interp_tiles, luts)  # Bx2GHx2GWx4xCx256

    # gh, gw -> 2x the number of tiles used to compute the histograms
    # th, tw -> /2 the sizes of the tiles used to compute the histograms
    num_imgs, gh, gw, c, th, tw = interp_tiles.shape

    # equalize tiles
    flatten_interp_tiles: torch.Tensor = (interp_tiles * 255).long().flatten(-2, -1)  # B x GH x GW x 4 x C x (THxTW)
    flatten_interp_tiles = flatten_interp_tiles.unsqueeze(-3).expand(num_imgs, gh, gw, 4, c, th * tw)
    preinterp_tiles_equalized = (
        torch.gather(mapped_luts, 5, flatten_interp_tiles)  # B x GH x GW x 4 x C x TH x TW
        .to(interp_tiles)
        .reshape(num_imgs, gh, gw, 4, c, th, tw)
    )

    # interp tiles
    tiles_equalized: torch.Tensor = torch.zeros_like(interp_tiles)

    # compute the interpolation weights (shapes are 2 x TH x TW because they must be applied to 2 interp tiles)
    ih = (
        torch.arange(2 * th - 1, -1, -1, dtype=interp_tiles.dtype, device=interp_tiles.device)
        .div(2.0 * th - 1)[None]
        .transpose(-2, -1)
        .expand(2 * th, tw)
    )
    ih = ih.unfold(0, th, th).unfold(1, tw, tw)  # 2 x 1 x TH x TW
    iw = (
        torch.arange(2 * tw - 1, -1, -1, dtype=interp_tiles.dtype, device=interp_tiles.device)
        .div(2.0 * tw - 1)
        .expand(th, 2 * tw)
    )
    iw = iw.unfold(0, th, th).unfold(1, tw, tw)  # 1 x 2 x TH x TW

    # compute row and column interpolation weights
    tiw = iw.expand((gw - 2) // 2, 2, th, tw).reshape(gw - 2, 1, th, tw).unsqueeze(0)  # 1 x GW-2 x 1 x TH x TW
    tih = ih.repeat((gh - 2) // 2, 1, 1, 1).unsqueeze(1)  # GH-2 x 1 x 1 x TH x TW

    # internal regions
    tl, tr, bl, br = preinterp_tiles_equalized[:, 1:-1, 1:-1].unbind(3)
    t = torch.addcmul(tr, tiw, torch.sub(tl, tr))
    b = torch.addcmul(br, tiw, torch.sub(bl, br))
    tiles_equalized[:, 1:-1, 1:-1] = torch.addcmul(b, tih, torch.sub(t, b))

    # corner regions
    tiles_equalized[:, 0 :: gh - 1, 0 :: gw - 1] = preinterp_tiles_equalized[:, 0 :: gh - 1, 0 :: gw - 1, 0]

    # border region (h)
    t, b, _, _ = preinterp_tiles_equalized[:, 1:-1, 0].unbind(2)
    tiles_equalized[:, 1:-1, 0] = torch.addcmul(b, tih.squeeze(1), torch.sub(t, b))
    t, b, _, _ = preinterp_tiles_equalized[:, 1:-1, gw - 1].unbind(2)
    tiles_equalized[:, 1:-1, gw - 1] = torch.addcmul(b, tih.squeeze(1), torch.sub(t, b))

    # border region (w)
    left, right, _, _ = preinterp_tiles_equalized[:, 0, 1:-1].unbind(2)
    tiles_equalized[:, 0, 1:-1] = torch.addcmul(right, tiw, torch.sub(left, right))
    left, right, _, _ = preinterp_tiles_equalized[:, gh - 1, 1:-1].unbind(2)
    tiles_equalized[:, gh - 1, 1:-1] = torch.addcmul(right, tiw, torch.sub(left, right))

    # same type as the input
    return tiles_equalized.div(255.0)


@perform_keep_shape_image
def equalize_clahe(
    input: torch.Tensor,
    clip_limit: float = 40.0,
    grid_size: Tuple[int, int] = (8, 8),
    slow_and_differentiable: bool = False,
) -> torch.Tensor:
    r"""Apply clahe equalization on the input tensor.

    .. image:: _static/img/equalize_clahe.png

    NOTE: Lut computation uses the same approach as in OpenCV, in next versions this can change.

    Args:
        input: images tensor to equalize with values in the range [0, 1] and shape :math:`(*, C, H, W)`.
        clip_limit: threshold value for contrast limiting. If 0 clipping is disabled.
        grid_size: number of tiles to be cropped in each direction (GH, GW). Each image axis must be larger
            than its grid size; otherwise a ``ValueError`` names the smallest image the grid admits.
        slow_and_differentiable: flag to select implementation

    Returns:
        Equalized image or images with shape as the input.

    Examples:
        >>> img = torch.rand(1, 10, 20)
        >>> res = equalize_clahe(img)
        >>> res.shape
        torch.Size([1, 10, 20])

        >>> img = torch.rand(2, 3, 10, 20)
        >>> res = equalize_clahe(img)
        >>> res.shape
        torch.Size([2, 3, 10, 20])

    .. note::
       The input is expected in :math:`[0, 1]`; each tile is equalized from a 256-bin lookup table.
       Values the lookup cannot index (outside roughly :math:`[0, 1]`) raise a ``RuntimeError``
       naming the range. The check is ``torch._assert_async``, which has an MPS kernel from torch
       ``2.13``; on an older MPS release the condition is read on the host instead, one device sync per
       call, and is skipped there under ``torch.compile``, as for :func:`kornia.enhance.equalize`.

    """
    if not isinstance(clip_limit, float):
        raise TypeError(f"Input clip_limit type is not float. Got {type(clip_limit)}")

    return _equalize_clahe(input, clip_limit, grid_size, slow_and_differentiable)


def _equalize_clahe(
    input: torch.Tensor,
    clip_limit: Union[float, torch.Tensor],
    grid_size: Tuple[int, int],
    slow_and_differentiable: bool,
) -> torch.Tensor:
    """Equalize a BCHW tensor using a scalar or one tensor clip limit per image."""
    if not isinstance(grid_size, tuple):
        raise TypeError(f"Input grid_size type is not Tuple. Got {type(grid_size)}")

    if len(grid_size) != 2:
        raise TypeError(f"Input grid_size is not a Tuple with 2 elements. Got {len(grid_size)}")

    if isinstance(grid_size[0], float) or isinstance(grid_size[1], float):
        raise TypeError("Input grid_size type is not valid, must be a Tuple[int, int].")

    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError(f"Input grid_size elements must be positive. Got {grid_size}")

    # The tile LUTs are gathered below with ``(interp_tiles * 255).long()``, which is in bounds
    # only for values in (-1/255, 256/255). Check that domain without ``.item()``, so there is no
    # device sync and fullgraph still compiles; on MPS before torch 2.13 it costs one host sync instead
    # (see ``_lookup_value_check``). Inputs the lookup can index are unchanged.
    _lookup_value_check(
        ((input * 255.0 > -1.0) & (input * 255.0 < 256.0)).all(),
        "equalize_clahe expects input values in [0, 1]. Scale the image into that range first, "
        "for example image / 255.0 for 8-bit data.",
    )

    imgs: torch.Tensor = input  # B x C x H x W

    # hist_tiles: torch.Tensor  # B x GH x GW x C x TH x TW  # not supported by JIT
    # img_padded: torch.Tensor  # B x C x H' x W'  # not supported by JIT
    # the size of the tiles must be even in order to divide them into 4 tiles for the interpolation
    hist_tiles, img_padded = _compute_tiles(imgs, grid_size, True)
    tile_size: Tuple[int, int] = (hist_tiles.shape[-2], hist_tiles.shape[-1])
    interp_tiles: torch.Tensor = _compute_interpolation_tiles(img_padded, tile_size)  # B x 2GH x 2GW x C x TH/2 x TW/2
    luts: torch.Tensor = _compute_luts(
        hist_tiles, clip=clip_limit, diff=slow_and_differentiable
    )  # B x GH x GW x C x 256
    equalized_tiles: torch.Tensor = _compute_equalized_tiles(interp_tiles, luts)  # B x 2GH x 2GW x C x TH/2 x TW/2

    # reconstruct the images form the tiles
    #    try permute + contiguous + view
    eq_imgs: torch.Tensor = equalized_tiles.permute(0, 3, 1, 4, 2, 5).reshape_as(img_padded)
    h, w = imgs.shape[-2:]
    eq_imgs = eq_imgs[..., :h, :w]  # crop imgs if they were padded

    # remove batch if the input was not in batch form
    if input.dim() != eq_imgs.dim():
        eq_imgs = eq_imgs.squeeze(0)

    return eq_imgs
