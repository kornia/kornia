"""Example of adaptative histogram equalization."""
from typing import List, Optional, Tuple, cast
import math
import time

from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import kornia
from kornia.utils import image


def plot_hist(lightness: torch.Tensor, title: Optional[str] = None) -> None:
    """Plot the histogram and the normalized cum histogram.

    Args:
      lightness (torch.Tensor): gray scale image (BxHxW)
      title (str, optional): title to be shown. Default: None

    """
    if lightness.dim() == 2:
        lightness = lightness[None]
    B = lightness.shape[0]
    vec: np.ndarray = lightness.mul(255).view(B, -1).cpu().numpy()

    fig, ax1 = plt.subplots(ncols=B)
    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.)
    ax = ax1
    for i in range(B):
        if B > 1:
            ax = ax1[i]
        color = "tab:blue"
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Histogram", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.hist(vec[i], range=(0, 255), bins=256, color=color)
        ax2 = ax.twinx()
        color = "tab:red"
        ax2.set_ylabel("Normalized Cumulative Histogram", color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.hist(vec[i], histtype="step", range=(0, 255), bins=256, cumulative=True, density=True, color=color)
    plt.tight_layout()
    plt.show()


def plot_image(img_rgb: torch.Tensor, lightness: Optional[torch.Tensor] = None) -> None:
    """Plot image changing the intensity.

    Args:
        img_rgb (torch.Tensor): original image (3xHxW) [0..1]
        lightness (torch.Tensor): normalized [0..1] intensity to be applied to each pixel (1xHxW).

    """
    img = img_rgb
    if img.dim() == 3:
        img = img[None]
    if lightness is not None:
        img_lab: torch.Tensor = kornia.rgb_to_lab(img)
        img_lab[..., 0, :, :] = lightness.mul(100).squeeze(-3)
        img = kornia.lab_to_rgb(img_lab)

    fig, ax = plt.subplots(ncols=img.shape[0])
    ax1 = ax
    for i in range(img.shape[0]):
        if img.shape[0] > 1:
            ax1 = ax[i]
        ax1.imshow(kornia.tensor_to_image(img[i].mul(255).clamp(0, 255).int()), cmap="gray")
        ax1.axis("off")
    plt.show()


def plot_hsv(img_hsv: torch.Tensor) -> None:
    """Plot image changing the intensity.

    Args:
        img_hsv (torch.Tensor): original image (WxHx3) [0..1]

    """
    img_rgb: torch.Tensor = kornia.hsv_to_rgb(img_hsv)
    plt.imshow(kornia.tensor_to_image(img_rgb.mul(255).clamp(0, 255).int()))
    plt.axis("off")


def visualize(tiles: torch.Tensor) -> None:
    """Show tiles as tiles.

    Args:
        tiles (torch.Tensor): set of tiles to be displayed (GH, GW, C, TH, TW)

    """
    fig = plt.figure(figsize=(tiles.shape[1], tiles.shape[0]))
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            inp = kornia.tensor_to_image(tiles[i][j])
            inp = np.array(inp)

            ax = fig.add_subplot(
                tiles.shape[0], tiles.shape[1], ((i * tiles.shape[1]) + j) + 1, xticks=[], yticks=[])
            plt.imshow(inp)
    plt.show()


def load_test_images(device: torch.device) -> torch.Tensor:
    """Load test images."""
    # load using opencv and convert to RGB
    list_img_rgb: List[torch.Tensor] = []
    img_bgr: np.ndarray = cv2.imread(
        "/Users/luis/Projects/kornia/examples/histogram_equalization/img1.png", cv2.IMREAD_COLOR)
    list_img_rgb.append(
        kornia.image_to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(dtype=torch.float32, device=device).div(255)
    )

    img_bgr = cv2.imread(
        "/Users/luis/Projects/kornia/examples/histogram_equalization/img2.jpg", cv2.IMREAD_COLOR)
    list_img_rgb.append(
        kornia.image_to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(dtype=torch.float32, device=device).div(255)
    )
    size: Tuple[int, int] = cast(Tuple[int, int], tuple([*list_img_rgb[0].shape[1:]]))
    list_img_rgb[1] = kornia.center_crop(list_img_rgb[1][None], size).squeeze()
    img_rgb: torch.Tensor = torch.stack(list_img_rgb)
    return img_rgb


def compute_tiles(imgs: torch.Tensor, grid_size: Tuple[int, int], even_tile_size: bool = False
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute tiles on an image according to a grid size.

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
    batch: torch.Tensor = image._to_bchw(imgs)  # B x C x H x W

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


def compute_interpolation_tiles(padded_imgs: torch.Tensor, tile_size: Tuple[int, int]) -> torch.Tensor:
    """Compute interpolation tiles on a properly padded set of images.

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
    interp_kernel_vert = tile_size[0] // 2
    interp_kernel_horz = tile_size[1] // 2

    c: int = padded_imgs.shape[-3]
    interp_tiles: torch.Tensor = (padded_imgs.unfold(1, c, c)
                                             .unfold(2, interp_kernel_vert, interp_kernel_vert)
                                             .unfold(3, interp_kernel_horz, interp_kernel_horz)
                                             .squeeze(1))  # 2GH x 2GW x C x TH/2 x TW/2
    assert interp_tiles.shape[-3] == c
    assert interp_tiles.shape[-2] == tile_size[0] / 2
    assert interp_tiles.shape[-1] == tile_size[1] / 2
    return interp_tiles


def compute_luts(tiles_x_im: torch.Tensor) -> torch.Tensor:
    """Compute luts for a batched set of tiles.

    Args:
        tiles_x_im (torch.Tensor): set of tiles per image to apply the lut. (B, GH, GW, C, TH, TW)

    Returns:
        torch.Tensor: Lut for each tile (B, GH, GW, C, 256)

    """
    def lut(patch: torch.Tensor, diff: bool = False) -> torch.Tensor:
        # function adapted from:
        # https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py
        # and https://github.com/pytorch/vision/pull/3119/files
        # and https://github.com/pytorch/vision/issues/1049
        # NOTE: torch.histc doesn't work with batches
        histo: torch.Tensor
        if not diff:
            histo = torch.histc(patch, bins=256, min=0, max=1)
        else:
            bins: torch.Tensor = torch.linspace(0, 1, 256, device=patch.device)
            histo = kornia.enhance.histogram(patch.flatten()[None], bins, torch.tensor(0.001)).squeeze()
            histo *= patch.shape[0] * patch.shape[1]

        nonzero_histo: torch.Tensor = histo[histo > 0.999]
        step: torch.Tensor
        if nonzero_histo.numel() > 0:
            step = (nonzero_histo.sum() - nonzero_histo[-1]) // 255
        else:
            step = torch.tensor(0, device=patch.device)
        if step == 0:
            return torch.zeros_like(histo).long()  # TODO: check the best return value for this case
        lut: torch.Tensor = (torch.cumsum(histo, 0) + (step // 2)) // step
        lut = torch.cat([torch.zeros(1, device=patch.device), lut[:-1]]).clamp(0, 255).long()
        return lut

    # precompute all the luts with 256 bins
    luts: torch.Tensor  # B x GH x GW x C x 256
    luts = torch.stack([torch.stack([torch.stack([torch.stack(
        [lut(c) for c in p]) for p in row_tiles]) for row_tiles in tiles]) for tiles in tiles_x_im])
    assert luts.shape == torch.Size([*tiles_x_im.shape[0:4]] + [256])
    return luts


def compute_luts_optim(tiles_x_im: torch.Tensor, num_bins: int = 256, clip: float = 40., diff: bool = False
                       ) -> torch.Tensor:
    """Compute luts for a batched set of tiles.

    Same approach as in OpenCV (https://github.com/opencv/opencv/blob/master/modules/imgproc/src/clahe.cpp)

    Args:
        tiles_x_im (torch.Tensor): set of tiles per image to apply the lut. (B, GH, GW, C, TH, TW)
        num_bins (int, optional): number of bins. default: 256
        clip (float): threshold value for contrast limiting. Default: 40
        diff (bool, optional): denote if the differentiable histagram will be used. Default: False

    Returns:
        torch.Tensor: Lut for each tile (B, GH, GW, C, 256)

    """
    pixels: int = tiles_x_im.shape[-2] * tiles_x_im.shape[-1]
    tiles: torch.Tensor = tiles_x_im.reshape(-1, pixels)  # test with view  # T x (THxTW)
    histos: torch.Tensor = torch.empty((tiles.shape[0], num_bins), device=tiles.device)
    if not diff:
        for i, tile in enumerate(tiles.unbind(0)):
            histos[i] = torch.histc(tile, bins=num_bins, min=0, max=1)
    else:
        bins: torch.Tensor = torch.linspace(0, 1, num_bins, device=tiles.device)
        histos = kornia.enhance.histogram(tiles, bins, torch.tensor(0.001)).squeeze()
        histos *= pixels

    # clip limit (TODO: optimice the code)
    if clip > 0:
        clip_limit = clip * pixels // num_bins
        clip_limit = max(clip_limit, 1)

        clip_idxs = histos > clip_limit
        for i, hist in enumerate(histos.unbind(0)):
            idxs = clip_idxs[i]
            if idxs.any():
                clipped = (hist[idxs] - clip_limit).sum()
                hist[idxs] = clip_limit

                redist = clipped // num_bins
                hist += redist

                residual = clipped - redist * num_bins
                if residual:
                    hist[0:int(residual)] += 1

    lut_scale = (num_bins - 1) / pixels
    luts = torch.cumsum(histos, 1) * lut_scale
    luts = luts.clamp(0, num_bins - 1).floor()  # to get the same values as converting to int maintaining the type
    luts = luts.view(([*tiles_x_im.shape[0:4]] + [num_bins]))
    return luts


def compute_equalized_tiles(interp_tiles: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    """Equalize the tiles.

    Args:
        interp_tiles (torch.Tensor): set of interpolation tiles. (B, 2GH, 2GW, C, TH/2, TW/2)
        luts (torch.Tensor): luts for each one of the original tiles. (B, GH, GW, C, 256)

    Returns:
        torvh.Tensor: equalized tiles (B, 2GH, 2GW, C, TH/2, TW/2)

    """
    tiles_equalized: torch.Tensor = torch.zeros_like(interp_tiles, dtype=torch.long)

    num_imgs: int  # number of batched images
    gh: int  # 2x the number of tiles used to compute the histograms
    gw: int
    c: int  # number of channels
    th: int  # /2 the sizes of the tiles used to compute the histograms
    tw: int
    num_imgs, gh, gw, c, th, tw = interp_tiles.shape

    # compute the interpolation weights (shapes are 2 x TH x TW because they must be applied to 2 interp tiles)
    ih = torch.arange(2 * th - 1, -1, -1, device=interp_tiles.device).div(2 * th - 1)[None].T.expand(2 * th, tw)
    ih = ih.unfold(0, th, th).unfold(1, tw, tw).squeeze(1)  # 2 x TH x TW
    iw = torch.arange(2 * tw - 1, -1, -1, device=interp_tiles.device).div(2 * tw - 1).expand(th, 2 * tw)
    iw = iw.unfold(0, th, th).unfold(1, tw, tw).squeeze(0)  # 2 x TH x TW
    # plot_image(m[0][None])
    # plot_image(n[0][None])

    flatten_interp_tiles: torch.Tensor = (interp_tiles * 255).long().flatten(-2, -1)  # B x GH x GW x C x (THxTW)
    for im in range(num_imgs):
        for j in range(gh):
            for i in range(gw):
                # corner region
                if (i == 0 or i == gw - 1) and (j == 0 or j == gh - 1):
                    a = torch.gather(luts[im, j // 2, i // 2], 1, flatten_interp_tiles[im, j, i])
                    a = a.reshape(c, th, tw)
                    tiles_equalized[im, j, i] = a
                    # print(f'corner ({j},{i})')
                    continue

                # border region (h)
                if i == 0 or i == gw - 1:
                    t = torch.gather(luts[im, max(0, j // 2 + j % 2 - 1), i // 2],
                                     1, flatten_interp_tiles[im, j, i]).reshape(c, th, tw)
                    b = torch.gather(luts[im, j // 2 + j % 2, i // 2],
                                     1, flatten_interp_tiles[im, j, i]).reshape(c, th, tw)
                    tiles_equalized[im, j, i] = ih[(j + 1) % 2] * (t - b) + b
                    # print(f'border h ({j},{i})')
                    continue

                # border region (w)
                if j == 0 or j == gh - 1:
                    l = torch.gather(luts[im, j // 2, max(0, i // 2 + i % 2 - 1)],
                                     1, flatten_interp_tiles[im, j, i]).reshape(c, th, tw)
                    r = torch.gather(luts[im, j // 2, i // 2 + i % 2],
                                     1, flatten_interp_tiles[im, j, i]).reshape(c, th, tw)
                    tiles_equalized[im, j, i] = iw[(i + 1) % 2] * (l - r) + r
                    # print(f'border w ({j},{i})')
                    continue

                # internal region
                tl = torch.gather(luts[im, max(0, j // 2 + j % 2 - 1), max(0, i // 2 + i % 2 - 1)],
                                  1, flatten_interp_tiles[im, j, i]).reshape(c, th, tw)
                tr = torch.gather(luts[im, max(0, j // 2 + j % 2 - 1), i // 2 + i % 2],
                                  1, flatten_interp_tiles[im, j, i]).reshape(c, th, tw)
                bl = torch.gather(luts[im, j // 2 + j % 2, max(0, i // 2 + i % 2 - 1)],
                                  1, flatten_interp_tiles[im, j, i]).reshape(c, th, tw)
                br = torch.gather(luts[im, j // 2 + j % 2, i // 2 + i % 2],
                                  1, flatten_interp_tiles[im, j, i]).reshape(c, th, tw)
                t = iw[(i + 1) % 2] * (tl - tr) + tr
                b = iw[(i + 1) % 2] * (bl - br) + br
                tiles_equalized[im, j, i] = ih[(j + 1) % 2] * (t - b) + b
    return tiles_equalized


def map_luts(interp_tiles: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    """Equalize the tiles.

    Args:
        interp_tiles (torch.Tensor): set of interpolation tiles. (B, 2GH, 2GW, C, TH/2, TW/2)
        luts (torch.Tensor): luts for each one of the original tiles. (B, GH, GW, C, 256)

    Returns:
        torch.Tensor: mapped luts (B, 2GH, 2GW, 4, C, 256)

    """
    num_imgs: int  # number of batched images
    gh: int  # 2x the number of tiles used to compute the histograms
    gw: int
    c: int  # number of channels
    th: int  # /2 the sizes of the tiles used to compute the histograms
    tw: int
    num_imgs, gh, gw, c, th, tw = interp_tiles.shape

    # precompute idxs for non corner regions
    # j_idxs = torch.zeros((gh - 2, 4), dtype=torch.long, device=luts.device)
    # i_idxs = torch.zeros((gw - 2, 4), dtype=torch.long, device=luts.device)
    # for j in range(1, gh - 1):
    #     v = j // 2 + j % 2
    #     j_idxs[j - 1] = torch.tensor([v - 1, v - 1, v, v], device=luts.device)
    # for i in range(1, gw - 1):
    #     v = i // 2 + i % 2
    #     i_idxs[i - 1] = torch.tensor([v - 1, v, v - 1, v], device=luts.device)

    # fast idxs (doing it in cpu seems sligthly faster)
    j_idxs = torch.ones(gh - 2, 4, dtype=torch.long) * torch.arange(1, gh - 1).reshape(gh - 2, 1)
    i_idxs = torch.ones(gw - 2, 4, dtype=torch.long) * torch.arange(1, gw - 1).reshape(gw - 2, 1)
    # j_idxs = torch.arange(1, gh - 1, device=interp_tiles.device).reshape(gh - 2, 1).repeat(1, 4)
    # i_idxs = torch.arange(1, gw - 1, device=interp_tiles.device).reshape(gh - 2, 1).repeat(1, 4)
    j_idxs = j_idxs // 2 + j_idxs % 2
    j_idxs[:, [0, 1]] -= 1
    i_idxs = i_idxs // 2 + i_idxs % 2
    i_idxs[:, [0, 2]] -= 1

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

    # t = luts_x_interp_tiles.clone()
    # for j in range(gh):
    #    for i in range(gw):
    #        # corner region
    #        if (i == 0 or i == gw - 1) and (j == 0 or j == gh - 1):
    #            luts_x_interp_tiles[:, j, i, 0] = luts[:, j // 2, i // 2]
    #            assert torch.allclose(luts_x_interp_tiles[:, j, i, :], t[:, j, i, :])
    #            continue

    #        # border region (h)
    #        if i == 0 or i == gw - 1:
    #            indexes = [max(0, j // 2 + j % 2 - 1), j // 2 + j % 2]
    #            luts_x_interp_tiles[:, j, i, [0, 1]] = luts[:, indexes, i // 2]
    #            assert torch.allclose(luts_x_interp_tiles[:, j, i, [0, 1]], t[:, j, i, [0, 1]])
    #            continue

    #        # border region (w)
    #        if j == 0 or j == gh - 1:
    #            indexes = [max(0, i // 2 + i % 2 - 1), i // 2 + i % 2]
    #            luts_x_interp_tiles[:, j, i, [0, 1]] = luts[:, j // 2, indexes]
    #            assert torch.allclose(luts_x_interp_tiles[:, j, i, :], t[:, j, i, :])
    #            continue

    #        # internal region
    #        j_indxs = [max(0, j // 2 + j % 2 - 1), max(0, j // 2 + j % 2 - 1), j // 2 + j % 2, j // 2 + j % 2]
    #        i_indxs = [max(0, i // 2 + i % 2 - 1), i // 2 + i % 2, max(0, i // 2 + i % 2 - 1), i // 2 + i % 2]
    #        luts_x_interp_tiles[:, j, i, :] = luts[:, j_indxs, i_indxs]
    #        assert torch.allclose(luts_x_interp_tiles[:, j, i, :], t[:, j, i, :])
    return luts_x_interp_tiles


def compute_equalized_tiles_opt(interp_tiles: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    """Equalize the tiles.

    Args:
        interp_tiles (torch.Tensor): set of interpolation tiles. (B, 2GH, 2GW, C, TH/2, TW/2)
        luts (torch.Tensor): luts for each one of the original tiles. (B, GH, GW, C, 256)

    Returns:
        torch.Tensor: equalized tiles (B, 2GH, 2GW, C, TH/2, TW/2)

    """
    mapped_luts: torch.Tensor = map_luts(interp_tiles, luts)  # Bx2GHx2GWx4xCx256

    num_imgs: int  # number of batched images
    gh: int  # 2x the number of tiles used to compute the histograms
    gw: int
    c: int  # number of channels
    th: int  # /2 the sizes of the tiles used to compute the histograms
    tw: int
    num_imgs, gh, gw, c, th, tw = interp_tiles.shape

    # equalize tiles
    flatten_interp_tiles: torch.Tensor = (interp_tiles * 255).long().flatten(-2, -1)  # B x GH x GW x 4 x C x (THxTW)
    flatten_interp_tiles = flatten_interp_tiles.unsqueeze(-3).expand(num_imgs, gh, gw, 4, c, th * tw)
    preinterp_tiles_equalized = torch.gather(
        mapped_luts, 5, flatten_interp_tiles).reshape(num_imgs, gh, gw, 4, c, th, tw)  # B x GH x GW x 4 x C x TH x TW

    # interp tiles
    tiles_equalized: torch.Tensor = torch.zeros_like(interp_tiles, dtype=torch.long)

    # compute the interpolation weights (shapes are 2 x TH x TW because they must be applied to 2 interp tiles)
    ih = torch.arange(2 * th - 1, -1, -1, device=interp_tiles.device).div(2 * th - 1)[None].T.expand(2 * th, tw)
    ih = ih.unfold(0, th, th).unfold(1, tw, tw)  # 2 x 1 x TH x TW
    iw = torch.arange(2 * tw - 1, -1, -1, device=interp_tiles.device).div(2 * tw - 1).expand(th, 2 * tw)
    iw = iw.unfold(0, th, th).unfold(1, tw, tw)  # 1 x 2 x TH x TW

    # compute row and column interpolation weigths
    tiw = iw.expand((gw - 2) // 2, 2, th, tw).reshape(gw - 2, 1, th, tw).unsqueeze(0)  # 1 x GW-2 x 1 x TH x TW
    tih = ih.repeat((gh - 2) // 2, 1, 1, 1).unsqueeze(1)  # GH-2 x 1 x 1 x TH x TW

    # internal regions
    tl, tr, bl, br = preinterp_tiles_equalized[:, 1:-1, 1:-1].unbind(3)
    # tl = preinterp_tiles_equalized[:, 1:-1, 1:-1, 0]
    # tr = preinterp_tiles_equalized[:, 1:-1, 1:-1, 1]
    # bl = preinterp_tiles_equalized[:, 1:-1, 1:-1, 2]
    # br = preinterp_tiles_equalized[:, 1:-1, 1:-1, 3]
    t = tiw * (tl - tr) + tr
    b = tiw * (bl - br) + br
    tiles_equalized[:, 1:-1, 1:-1] = tih * (t - b) + b

    # corner regions
    tiles_equalized[:, 0::gh - 1, 0::gw - 1] = preinterp_tiles_equalized[:, 0::gh - 1, 0::gw - 1, 0]
    # tiles_equalized[:, 0, 0] = preinterp_tiles_equalized[:, 0, 0, 0]
    # tiles_equalized[:, gh - 1, 0] = preinterp_tiles_equalized[:, gh - 1, 0, 0]
    # tiles_equalized[:, 0, gw - 1] = preinterp_tiles_equalized[:, 0, gw - 1, 0]
    # tiles_equalized[:, gh - 1, gw - 1] = preinterp_tiles_equalized[:, gh - 1, gw - 1, 0]

    # border region (h)
    t, b, _, _ = preinterp_tiles_equalized[:, 1:-1, 0].unbind(2)
    # t = preinterp_tiles_equalized[:, 1:-1, 0, 0]
    # b = preinterp_tiles_equalized[:, 1:-1, 0, 1]
    tiles_equalized[:, 1:-1, 0] = tih.squeeze(1) * (t - b) + b

    t, b, _, _ = preinterp_tiles_equalized[:, 1:-1, gh - 1].unbind(2)
    # t = preinterp_tiles_equalized[:, 1:-1, gh - 1, 0]
    # b = preinterp_tiles_equalized[:, 1:-1, gh - 1, 1]
    tiles_equalized[:, 1:-1, gh - 1] = tih.squeeze(1) * (t - b) + b

    # border region (w)
    l, r, _, _ = preinterp_tiles_equalized[:, 0, 1:-1].unbind(2)
    # l = preinterp_tiles_equalized[:, 0, 1:-1, 0]
    # r = preinterp_tiles_equalized[:, 0, 1:-1, 1]
    tiles_equalized[:, 0, 1:-1] = tiw * (l - r) + r

    l, r, _, _ = preinterp_tiles_equalized[:, gw - 1, 1:-1].unbind(2)
    # l = preinterp_tiles_equalized[:, gw - 1, 1:-1, 0]
    # r = preinterp_tiles_equalized[:, gw - 1, 1:-1, 1]
    tiles_equalized[:, gw - 1, 1:-1] = tiw * (l - r) + r

    return tiles_equalized


def equalize_clahe(input: torch.Tensor, clip_limit: float = 40., grid_size: Tuple[int, int] = (8, 8)) -> torch.Tensor:
    r"""Apply clahe equalization on the input tensor.

    Args:
        input (torch.Tensor): images tensor to equalize with shapes like :math:`(C, H, W)` or :math:`(B, C, H, W)`.
        clip_limit (float): threshold value for contrast limiting. Default: 40.
        grid_size (Tuple[int, int]): number of tiles to be cropped in each direction (GH, GW).

    Returns:
        torch.Tensor: Sharpened image or images with shape as the input.

    """
    imgs: torch.Tensor = image._to_bchw(input)  # B x C x H x W

    hist_tiles: torch.Tensor  # B x GH x GW x C x TH x TW
    img_padded: torch.Tensor  # B x C x H' x W'
    # the size of the tiles must be even in order to divide them into 4 tiles for the interpolation
    hist_tiles, img_padded = compute_tiles(imgs, grid_size, True)
    tile_size: Tuple[int, int] = hist_tiles.shape[-2:]  # type: ignore
    interp_tiles: torch.Tensor = (
        compute_interpolation_tiles(img_padded, tile_size))  # B x 2GH x 2GW x C x TH/2 x TW/2
    luts: torch.Tensor = compute_luts_optim(hist_tiles, clip=clip_limit)  # B x GH x GW x C x B
    equalized_tiles: torch.Tensor = compute_equalized_tiles_opt(interp_tiles, luts)  # B x 2GH x 2GW x C x TH/2 x TW/2

    # reconstruct the images form the tiles
    eq_imgs: torch.Tensor = torch.cat(equalized_tiles.unbind(2), 4)
    eq_imgs = torch.cat(eq_imgs.unbind(1), 2)
    h, w = imgs.shape[-2:]
    eq_imgs = eq_imgs[..., :h, :w]  # crop imgs if they were padded

    # remove batch if the input was not in batch form
    if input.dim() != eq_imgs.dim():
        eq_imgs.squeeze_(0)
    return eq_imgs


def main():
    """Run the main function."""
    clip_limit = 2.
    on_rgb: bool = False
    if not torch.cuda.is_available():
        print("WARNING: Cuda is not enabled!!!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_rgb: torch.Tensor = load_test_images(device)  # B x C x H x W
    img: torch.Tensor = img_rgb
    if not on_rgb:
        img_lab: torch.Tensor = kornia.rgb_to_lab(img_rgb)
        img = img_lab[..., 0, :, :].unsqueeze(-3) / 100  # L in lab is in range [0, 100]
    # plot_image(img_rgb)
    gh = gw = 8
    grid_size: Tuple = (gh, gw)
    hist_tiles: torch.Tensor  # B x GH x GW x C x TH x TW
    img_padded: torch.Tensor  # B x C x H' x W'
    # the size of the tiles must be even in order to divide them into 4 tiles for the interpolation
    tic = time.time()
    hist_tiles, img_padded = compute_tiles(img, grid_size, True)
    # print(hist_tiles.shape)
    # visualize(hist_tiles[0])
    # visualize(hist_tiles[1])
    tile_size: Tuple = hist_tiles.shape[-2:]
    interp_tiles: torch.Tensor = (
        compute_interpolation_tiles(img_padded, tile_size))  # B x 2GH x 2GW x C x TH/2 x TW/2
    # print(interp_tiles.shape)
    # visualize(interp_tiles[0])
    # visualize(interp_tiles[1])
    time_tiles = time.time() - tic

    # for i in range(10):
    #    luts: torch.Tensor = compute_luts_optim(hist_tiles, clip=clip_limit)  # B x GH x GW x C x B
    #    equalized_tiles: torch.Tensor = compute_equalized_tiles_opt(interp_tiles, luts)  # B x 2GH x 2GW x C x TH/2 x TW/2

    #p1 = torch.cat(equalized_tiles.unbind(2), 4)
    #p2 = torch.cat(p1.unbind(1), 2)
    #h, w = img_rgb.shape[-2:]
    #p2 = p2[..., :h, :w]

    tic = time.time()
    p2 = equalize_clahe(img, clip_limit, (gh, gw))
    time_my_clahe = time.time() - tic

#    if on_rgb:
#        plot_image(p2.div(255.))
#    else:
#        plot_image(img_rgb, p2.div(255.))
#        plot_hist(p2.div(255.))

    tic = time.time()
    for i in range(10):
        # luts = compute_luts(hist_tiles)  # B x GH x GW x C x B
        # equalized_tiles: torch.Tensor = compute_equalized_tiles(interp_tiles, luts)  # B x 2GH x 2GW x C x TH/2 x TW/2
        luts: torch.Tensor = compute_luts_optim(hist_tiles, clip=0.)  # B x GH x GW x C x B
        equalized_tiles: torch.Tensor = compute_equalized_tiles_opt(interp_tiles, luts)  # B x 2GH x 2GW x C x TH/2 x TW/2

    p1 = torch.cat(equalized_tiles.unbind(2), 4)
    p2 = torch.cat(p1.unbind(1), 2)
    h, w = img_rgb.shape[-2:]
    p2 = p2[..., :h, :w]
    time_my_ahe = time.time() - tic

#    if on_rgb:
#        plot_image(p2.div(255.))
#    else:
#        plot_image(img_rgb, p2.div(255.))
#        plot_hist(p2.div(255.))

    # hist equalization in kornia
    tic = time.time()
    for i in range(10):
        lightness_equalized = kornia.enhance.equalize(img).squeeze()
    time_kornia_he = time.time() - tic
#    plot_image(img_rgb, lightness_equalized)
#    plot_hist(lightness_equalized)

    # hist equalization in opencv
    ims = img.mul(255).clamp(0, 255).byte().cpu().numpy().squeeze()
    tic = time.time()
    for i in range(10):
        equ0 = cv2.equalizeHist(ims[0])
        equ1 = cv2.equalizeHist(ims[1])
    time_opencv_he = time.time() - tic
#    plot_hist(torch.tensor([equ0, equ1]).float().div(255))
#    plot_image(img_rgb, torch.tensor([equ0, equ1]).float().div(255))

    ims = img.mul(255).clamp(0, 255).byte().cpu().numpy()
    tic = time.time()
    for i in range(10):
        # with this clip limit produces the "same" result as ahe
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(gh, gw))
        res = np.zeros_like(ims)
        for j in range(ims.shape[1]):
            res[0, j] = clahe.apply(ims[0][j])
            res[1, j] = clahe.apply(ims[1][j])
    time_opencv_clahe = time.time() - tic
#    if on_rgb:
#        plot_image(torch.tensor(res).float().div(255))
#    else:
#        plot_hist(torch.tensor(res).float().div(255))
#        plot_image(img_rgb, torch.tensor(res).float().div(255))

    print(f'time_tiles: \t{time_tiles:.5f}\nmy clahe: \t{time_my_clahe:.5f}\nmy ahe: \t{time_my_ahe:.5f}\nkornia he: \t{time_kornia_he:.5f}\nopencv he: \t{time_opencv_he:.5f}\nopencv clahe: \t{time_opencv_clahe:.5f}')


if __name__ == "__main__":
    main()
