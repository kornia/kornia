import math

import torch
import torch.nn as nn


def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: HSV version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)
=======
<<<<<<< master
<<<<<<< refs/remotes/kornia/master
>>>>>>> [Feat] refactor tests for kornia.color (#759)
=======
<<<<<<< master
<<<<<<< refs/remotes/kornia/master
>>>>>>> fix few jit and cuda errors in color (#767)
=======
<<<<<<< master
=======
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)
    # The first or last occurance is not guarenteed before 1.6.0
    # https://github.com/pytorch/pytorch/issues/20414
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
    maxc, max_indices = image.max(-3)
>>>>>>> [Feat] refactor tests for kornia.color (#759)
=======
    # TODO: enable again for later versions than 1.6.0 or find a different implementation.
    # It turns out that .max(...) does not return the index in the first position when
    # all the inputs have the same value in CUDA.
    # maxc, max_indices = image.max(-3)
    if image.is_cuda and torch.__version__ == '1.6.0':
        maxc, max_indices = image.cpu().max(-3)
        maxc, max_indices = maxc.to(image), max_indices.to(image.device)
    else:
        maxc, max_indices = image.max(-3)

>>>>>>> fix few jit and cuda errors in color (#767)
=======
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)
=======
=======
<<<<<<< master
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)
=======
    maxc, max_indices = image.max(-3)
>>>>>>> [Feat] refactor tests for kornia.color (#759)
<<<<<<< refs/remotes/kornia/master
>>>>>>> [Feat] refactor tests for kornia.color (#759)
=======
=======
    # TODO: enable again for later versions than 1.6.0 or find a different implementation.
    # It turns out that .max(...) does not return the index in the first position when
    # all the inputs have the same value in CUDA.
    # maxc, max_indices = image.max(-3)
    if image.is_cuda and torch.__version__ == '1.6.0':
        maxc, max_indices = image.cpu().max(-3)
        maxc, max_indices = maxc.to(image), max_indices.to(image.device)
    else:
        maxc, max_indices = image.max(-3)

>>>>>>> fix few jit and cuda errors in color (#767)
<<<<<<< refs/remotes/kornia/master
>>>>>>> fix few jit and cuda errors in color (#767)
=======
=======
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / (v + 1e-31)

    # avoid division by zero
    deltac = torch.where(
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
<<<<<<< master
<<<<<<< refs/remotes/kornia/master
>>>>>>> [Feat] refactor tests for kornia.color (#759)
=======
<<<<<<< master
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)
        deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]
=======
        deltac == 0, torch.ones_like(deltac), deltac)
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)
=======
        deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)
>>>>>>> [Feat] Enabled Torch1.5.1 cpu support (#796)

<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
<<<<<<< master
>>>>>>> fix few jit and cuda errors in color (#767)
    rc, gc, bc = torch.unbind(maxc.unsqueeze(-3) - image, dim=-3)
>>>>>>> [Feat] refactor tests for kornia.color (#759)
=======
    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]
>>>>>>> fix few jit and cuda errors in color (#767)
<<<<<<< refs/remotes/kornia/master
=======

    rc, gc, bc = torch.unbind(maxc.unsqueeze(-3) - image, dim=-3)
>>>>>>> [Feat] refactor tests for kornia.color (#759)
>>>>>>> [Feat] refactor tests for kornia.color (#759)
=======
>>>>>>> fix few jit and cuda errors in color (#767)

    h = torch.stack([
        bc - gc,
        2.0 * deltac + rc - bc,
        4.0 * deltac + gc - rc,
    ], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.).to(image.device)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((
        v, q, p, p, t, v,
        t, v, v, q, p, p,
        p, p, t, v, v, q,
    ), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


class RgbToHsv(nn.Module):
    r"""Convert an image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.tensor: HSV version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hsv = RgbToHsv()
        >>> output = hsv(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(RgbToHsv, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_hsv(image)


class HsvToRgb(nn.Module):
    r"""Convert an image from HSV to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HsvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """
=======
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    maxc, max_indices = image.max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / (v + 1e-31)

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    rc, gc, bc = torch.unbind(maxc.unsqueeze(-3) - image, dim=-3)
=======
        torch.Tensor: RGB version of the image.
>>>>>>> [Feat] refactor tests for kornia.color (#759)

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

<<<<<<< refs/remotes/kornia/master
    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac
>>>>>>> Accelerate augmentations (#708)
=======
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HsvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """
>>>>>>> [Feat] refactor tests for kornia.color (#759)

    def __init__(self) -> None:
        super(HsvToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return hsv_to_rgb(image)
