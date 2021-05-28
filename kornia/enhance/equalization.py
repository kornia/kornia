"""In this module several equalization methods are exposed: he, ahe, clahe."""

from typing import Tuple
import math

import torch
import torch.nn.functional as F

from kornia.enhance.histogram import histogram
from kornia.utils.image import _to_bchw
from kornia.utils.helpers import _torch_histc_cast


__all__ = ["equalize_clahe"]


def _compute_tiles(imgs: torch.Tensor, grid_size: Tuple[int, int], even_tile_size: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute tiles on an image according to a grid size.

    Note that padding can be added to the image in order to crop properly the image.
    So, the grid_size (GH, GW) x tile_size (TH, TW) >= image_size (H, W)

    Args:
        imgs (torch.Tensor): batch of 2D images with shape (B, C, H, W) or (C, H, W).
        grid_size (Tuple[int, int]): number of tiles to be cropped in each direction (GH, GW)
        even_tile_size (bool, optional): Determine if the width and height of the tiles must be even. Default: False.

    Returns:
        torch.Tensor: tensor with tiles (B, GH, GW, C, TH, TW). B = 1 in case of a single image is provided.
        torch.Tensor: tensor with the padded batch of 2D imageswith shape (B, C, H', W')

    """
    batch: torch.Tensor = _to_bchw(imgs)  # B x C x H x W

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
    if pad_vert > 0 or pad_horz > 0:
        batch = F.pad(batch, [0, pad_horz, 0, pad_vert], mode='reflect')  # B x C x H' x W'

    # compute tiles
    c: int = batch.shape[-3]
    tiles: torch.Tensor = (batch.unfold(1, c, c)  # unfold(dimension, size, step)
                                .unfold(2, kernel_vert, kernel_vert)
                                .unfold(3, kernel_horz, kernel_horz)
                                .squeeze(1))  # GH x GW x C x TH x TW
    assert tiles.shape[-5] == grid_size[0]  # check the grid size
    assert tiles.shape[-4] == grid_size[1]
    return tiles, batch


def _compute_interpolation_tiles(padded_imgs: torch.Tensor, tile_size: Tuple[int, int]) -> torch.Tensor:
    r"""Compute interpolation tiles on a properly padded set of images.

    Note that images must be padded. So, the tile_size (TH, TW) * grid_size (GH, GW) = image_size (H, W)

    Args:
        padded_imgs (torch.Tensor): batch of 2D images with shape (B, C, H, W) already padded to extract tiles
                                    of size (TH, TW).
        tile_size (Tuple[int, int]): shape of the current tiles (TH, TW).

    Returns:
        torch.Tensor: tensor with the interpolation tiles (B, 2GH, 2GW, C, TH/2, TW/2).

    """
    assert padded_imgs.dim() == 4, "Images Tensor must be 4D."
    assert padded_imgs.shape[-2] % tile_size[0] == 0, "Images are not correctly padded."
    assert padded_imgs.shape[-1] % tile_size[1] == 0, "Images are not correctly padded."

    # tiles to be interpolated are built by dividing in 4 each alrady existing
    interp_kernel_vert: int = tile_size[0] // 2
    interp_kernel_horz: int = tile_size[1] // 2

    c: int = padded_imgs.shape[-3]
    interp_tiles: torch.Tensor = (padded_imgs.unfold(1, c, c)
                                             .unfold(2, interp_kernel_vert, interp_kernel_vert)
                                             .unfold(3, interp_kernel_horz, interp_kernel_horz)
                                             .squeeze(1))  # 2GH x 2GW x C x TH/2 x TW/2
    assert interp_tiles.shape[-3] == c
    assert interp_tiles.shape[-2] == tile_size[0] / 2
    assert interp_tiles.shape[-1] == tile_size[1] / 2
    return interp_tiles


def _compute_luts(tiles_x_im: torch.Tensor, num_bins: int = 256, clip: float = 40., diff: bool = False) -> torch.Tensor:
    r"""Compute luts for a batched set of tiles.

    Same approach as in OpenCV (https://github.com/opencv/opencv/blob/master/modules/imgproc/src/clahe.cpp)

    Args:
        tiles_x_im (torch.Tensor): set of tiles per image to apply the lut. (B, GH, GW, C, TH, TW)
        num_bins (int, optional): number of bins. default: 256
        clip (float): threshold value for contrast limiting. If it is 0 then the clipping is disabled. Default: 40.
        diff (bool, optional): denote if the differentiable histagram will be used. Default: False

    Returns:
        torch.Tensor: Lut for each tile (B, GH, GW, C, 256)

    """
    assert tiles_x_im.dim() == 6, "Tensor must be 6D."

    b, gh, gw, c, th, tw = tiles_x_im.shape
    pixels: int = th * tw
    tiles: torch.Tensor = tiles_x_im.reshape(-1, pixels)  # test with view  # T x (THxTW)
    histos: torch.Tensor = torch.empty((tiles.shape[0], num_bins), device=tiles.device)
    if not diff:
        for i in range(tiles.shape[0]):
            histos[i] = _torch_histc_cast(tiles[i], bins=num_bins, min=0, max=1)
    else:
        bins: torch.Tensor = torch.linspace(0, 1, num_bins, device=tiles.device)
        histos = histogram(tiles, bins, torch.tensor(0.001)).squeeze()
        histos *= pixels

    # clip limit (TODO: optimice the code)
    if clip > 0.:
        clip_limit: torch.Tensor = torch.tensor(
            max(clip * pixels // num_bins, 1), dtype=histos.dtype, device=tiles.device)

        clip_idxs: torch.Tensor = histos > clip_limit
        for i in range(histos.shape[0]):
            hist: torch.Tensor = histos[i]
            idxs = clip_idxs[i]
            if idxs.any():
                clipped: float = float((hist[idxs] - clip_limit).sum().item())
                hist = torch.where(idxs, clip_limit, hist)

                redist: float = clipped // num_bins
                hist += redist

                residual: float = clipped - redist * num_bins
                if residual:
                    hist[0:int(residual)] += 1
                histos[i] = hist

    lut_scale: float = (num_bins - 1) / pixels
    luts: torch.Tensor = torch.cumsum(histos, 1) * lut_scale
    luts = luts.clamp(0, num_bins - 1).floor()  # to get the same values as converting to int maintaining the type
    luts = luts.view((b, gh, gw, c, num_bins))
    return luts


def _map_luts(interp_tiles: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    r"""Assign the required luts to each tile.

    Args:
        interp_tiles (torch.Tensor): set of interpolation tiles. (B, 2GH, 2GW, C, TH/2, TW/2)
        luts (torch.Tensor): luts for each one of the original tiles. (B, GH, GW, C, 256)

    Returns:
        torch.Tensor: mapped luts (B, 2GH, 2GW, 4, C, 256)

    """
    assert interp_tiles.dim() == 6, "interp_tiles tensor must be 6D."
    assert luts.dim() == 5, "luts tensor must be 5D."

    # gh, gw -> 2x the number of tiles used to compute the histograms
    # th, tw -> /2 the sizes of the tiles used to compute the histograms
    num_imgs, gh, gw, c, th, tw = interp_tiles.shape

    # precompute idxs for non corner regions (doing it in cpu seems sligthly faster)
    j_idxs = torch.ones(gh - 2, 4, dtype=torch.long) * torch.arange(1, gh - 1).reshape(gh - 2, 1)
    i_idxs = torch.ones(gw - 2, 4, dtype=torch.long) * torch.arange(1, gw - 1).reshape(gw - 2, 1)
    j_idxs = j_idxs // 2 + j_idxs % 2
    j_idxs[:, 0:2] -= 1
    i_idxs = i_idxs // 2 + i_idxs % 2
    # i_idxs[:, [0, 2]] -= 1  # this slicing is not supported by jit
    i_idxs[:, 0] -= 1
    i_idxs[:, 2] -= 1

    # selection of luts to interpolate each patch
    # create a tensor with dims: interp_patches height and width x 4 x num channels x bins in the histograms
    # the tensor is init to -1 to denote non init hists
    luts_x_interp_tiles: torch.Tensor = -torch.ones(
        num_imgs, gh, gw, 4, c, luts.shape[-1], device=interp_tiles.device)  # B x GH x GW x 4 x C x 256
    # corner regions
    luts_x_interp_tiles[:, 0::gh - 1, 0::gw - 1, 0] = luts[:, 0::max(gh // 2 - 1, 1), 0::max(gw // 2 - 1, 1)]
    # border region (h)
    luts_x_interp_tiles[:, 1:-1, 0::gw - 1, 0] = luts[:, j_idxs[:, 0], 0::max(gw // 2 - 1, 1)]
    luts_x_interp_tiles[:, 1:-1, 0::gw - 1, 1] = luts[:, j_idxs[:, 2], 0::max(gw // 2 - 1, 1)]
    # border region (w)
    luts_x_interp_tiles[:, 0::gh - 1, 1:-1, 0] = luts[:, 0::max(gh // 2 - 1, 1), i_idxs[:, 0]]
    luts_x_interp_tiles[:, 0::gh - 1, 1:-1, 1] = luts[:, 0::max(gh // 2 - 1, 1), i_idxs[:, 1]]
    # internal region
    luts_x_interp_tiles[:, 1:-1, 1:-1, :] = luts[
        :, j_idxs.repeat(max(gh - 2, 1), 1, 1).permute(1, 0, 2), i_idxs.repeat(max(gw - 2, 1), 1, 1)]

    return luts_x_interp_tiles


def _compute_equalized_tiles(interp_tiles: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    r"""Equalize the tiles.

    Args:
        interp_tiles (torch.Tensor): set of interpolation tiles, values must be in the range [0, 1].
                                     (B, 2GH, 2GW, C, TH/2, TW/2)
        luts (torch.Tensor): luts for each one of the original tiles. (B, GH, GW, C, 256)

    Returns:
        torch.Tensor: equalized tiles (B, 2GH, 2GW, C, TH/2, TW/2)

    """
    assert interp_tiles.dim() == 6, "interp_tiles tensor must be 6D."
    assert luts.dim() == 5, "luts tensor must be 5D."

    mapped_luts: torch.Tensor = _map_luts(interp_tiles, luts)  # Bx2GHx2GWx4xCx256

    # gh, gw -> 2x the number of tiles used to compute the histograms
    # th, tw -> /2 the sizes of the tiles used to compute the histograms
    num_imgs, gh, gw, c, th, tw = interp_tiles.shape

    # equalize tiles
    flatten_interp_tiles: torch.Tensor = (interp_tiles * 255).long().flatten(-2, -1)  # B x GH x GW x 4 x C x (THxTW)
    flatten_interp_tiles = flatten_interp_tiles.unsqueeze(-3).expand(num_imgs, gh, gw, 4, c, th * tw)
    preinterp_tiles_equalized = torch.gather(
        mapped_luts, 5, flatten_interp_tiles).reshape(num_imgs, gh, gw, 4, c, th, tw)  # B x GH x GW x 4 x C x TH x TW

    # interp tiles
    tiles_equalized: torch.Tensor = torch.zeros_like(interp_tiles, dtype=torch.long)

    # compute the interpolation weights (shapes are 2 x TH x TW because they must be applied to 2 interp tiles)
    ih = torch.arange(2 * th - 1, -1, -1, device=interp_tiles.device).div(
        2. * th - 1)[None].transpose(-2, -1).expand(2 * th, tw)
    ih = ih.unfold(0, th, th).unfold(1, tw, tw)  # 2 x 1 x TH x TW
    iw = torch.arange(2 * tw - 1, -1, -1, device=interp_tiles.device).div(2. * tw - 1).expand(th, 2 * tw)
    iw = iw.unfold(0, th, th).unfold(1, tw, tw)  # 1 x 2 x TH x TW

    # compute row and column interpolation weigths
    tiw = iw.expand((gw - 2) // 2, 2, th, tw).reshape(gw - 2, 1, th, tw).unsqueeze(0)  # 1 x GW-2 x 1 x TH x TW
    tih = ih.repeat((gh - 2) // 2, 1, 1, 1).unsqueeze(1)  # GH-2 x 1 x 1 x TH x TW

    # internal regions
    tl, tr, bl, br = preinterp_tiles_equalized[:, 1:-1, 1:-1].unbind(3)
    t = tiw * (tl - tr) + tr
    b = tiw * (bl - br) + br
    tiles_equalized[:, 1:-1, 1:-1] = tih * (t - b) + b

    # corner regions
    tiles_equalized[:, 0::gh - 1, 0::gw - 1] = preinterp_tiles_equalized[:, 0::gh - 1, 0::gw - 1, 0]

    # border region (h)
    t, b, _, _ = preinterp_tiles_equalized[:, 1:-1, 0].unbind(2)
    tiles_equalized[:, 1:-1, 0] = tih.squeeze(1) * (t - b) + b
    t, b, _, _ = preinterp_tiles_equalized[:, 1:-1, gh - 1].unbind(2)
    tiles_equalized[:, 1:-1, gh - 1] = tih.squeeze(1) * (t - b) + b

    # border region (w)
    l, r, _, _ = preinterp_tiles_equalized[:, 0, 1:-1].unbind(2)
    tiles_equalized[:, 0, 1:-1] = tiw * (l - r) + r
    l, r, _, _ = preinterp_tiles_equalized[:, gw - 1, 1:-1].unbind(2)
    tiles_equalized[:, gw - 1, 1:-1] = tiw * (l - r) + r

    # same type as the input
    return tiles_equalized.to(interp_tiles).div(255.)


def equalize_clahe(input: torch.Tensor, clip_limit: float = 40., grid_size: Tuple[int, int] = (8, 8)) -> torch.Tensor:
    r"""Apply clahe equalization on the input tensor.

    NOTE: Lut computation uses the same approach as in OpenCV, in next versions this can change.

    Args:
        input (torch.Tensor): images tensor to equalize with values in the range [0, 1] and shapes like
                              :math:`(C, H, W)` or :math:`(B, C, H, W)`.
        clip_limit (float): threshold value for contrast limiting. If 0 clipping is disabled. Default: 40.
        grid_size (Tuple[int, int]): number of tiles to be cropped in each direction (GH, GW). Default: (8, 8).

    Returns:
        torch.Tensor: Equalized image or images with shape as the input.

    Examples:
        >>> img = torch.rand(1, 10, 20)
        >>> res = equalize_clahe(img)
        >>> res.shape
        torch.Size([1, 10, 20])

        >>> img = torch.rand(2, 3, 10, 20)
        >>> res = equalize_clahe(img)
        >>> res.shape
        torch.Size([2, 3, 10, 20])

    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input type is not a torch.Tensor. Got {type(input)}")

    if input.dim() not in [3, 4]:
        raise ValueError(f"Invalid input shape, we expect CxHxW or BxCxHxW. Got: {input.shape}")

    if input.numel() == 0:
        raise ValueError("Invalid input tensor, it is empty.")

    if not isinstance(clip_limit, float):
        raise TypeError(f"Input clip_limit type is not float. Got {type(clip_limit)}")

    if not isinstance(grid_size, tuple):
        raise TypeError(f"Input grid_size type is not Tuple. Got {type(grid_size)}")

    if len(grid_size) != 2:
        raise TypeError(f"Input grid_size is not a Tuple with 2 elements. Got {len(grid_size)}")

    if isinstance(grid_size[0], float) or isinstance(grid_size[1], float):
        raise TypeError("Input grid_size type is not valid, must be a Tuple[int, int].")

    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError("Input grid_size elements must be positive. Got {grid_size}")

    imgs: torch.Tensor = _to_bchw(input)  # B x C x H x W

    # hist_tiles: torch.Tensor  # B x GH x GW x C x TH x TW  # not supported by JIT
    # img_padded: torch.Tensor  # B x C x H' x W'  # not supported by JIT
    # the size of the tiles must be even in order to divide them into 4 tiles for the interpolation
    hist_tiles, img_padded = _compute_tiles(imgs, grid_size, True)
    tile_size: Tuple[int, int] = (hist_tiles.shape[-2], hist_tiles.shape[-1])
    interp_tiles: torch.Tensor = (
        _compute_interpolation_tiles(img_padded, tile_size))  # B x 2GH x 2GW x C x TH/2 x TW/2
    luts: torch.Tensor = _compute_luts(hist_tiles, clip=clip_limit)  # B x GH x GW x C x B
    equalized_tiles: torch.Tensor = _compute_equalized_tiles(interp_tiles, luts)  # B x 2GH x 2GW x C x TH/2 x TW/2

    # reconstruct the images form the tiles
    eq_imgs: torch.Tensor = torch.cat(equalized_tiles.unbind(2), 4)
    eq_imgs = torch.cat(eq_imgs.unbind(1), 2)
    h, w = imgs.shape[-2:]
    eq_imgs = eq_imgs[..., :h, :w]  # crop imgs if they were padded

    # remove batch if the input was not in batch form
    if input.dim() != eq_imgs.dim():
        eq_imgs = eq_imgs.squeeze(0)
    return eq_imgs
