import math

import torch
import torch.nn as nn


def rgb_to_hls(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert a RGB image to HLS.

    .. image:: _static/img/rgb_to_hls.png

    The image data is assumed to be in the range of (0, 1).

    NOTE: this method cannot be compiled with JIT in pytohrch < 1.7.0

    Args:
        image: RGB image to be converted to HLS with shape :math:`(*, 3, H, W)`.
        eps: epsilon value to avoid div by zero.

    Returns:
        HLS version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hls(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if not torch.jit.is_scripting():
        # weird way to use globals compiling with JIT even in the code not used by JIT...
        # __setattr__ can be removed if pytorch version is > 1.6.0 and then use:
        # rgb_to_hls.RGB2HSL_IDX = hls_to_rgb.RGB2HSL_IDX.to(image.device)
        rgb_to_hls.__setattr__('RGB2HSL_IDX', rgb_to_hls.RGB2HSL_IDX.to(image))  # type: ignore
        _RGB2HSL_IDX: torch.Tensor = rgb_to_hls.RGB2HSL_IDX  # type: ignore
    else:
        _RGB2HSL_IDX = torch.tensor([[[0.0]], [[1.0]], [[2.0]]], device=image.device, dtype=image.dtype)  # 3x1x1

    # maxc: torch.Tensor  # not supported by JIT
    # imax: torch.Tensor  # not supported by JIT
    maxc, imax = image.max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    # h: torch.Tensor  # not supported by JIT
    # l: torch.Tensor  # not supported by JIT
    # s: torch.Tensor  # not supported by JIT
    # image_hls: torch.Tensor  # not supported by JIT
    if image.requires_grad:
        l = maxc + minc
        s = maxc - minc
        # weird behaviour with undefined vars in JIT...
        # scripting requires image_hls be defined even if it is not used :S
        h = l  # assign to any tensor...
        image_hls = l  # assign to any tensor...
    else:
        # define the resulting image to avoid the torch.stack([h, l, s])
        # so, h, l and s require inplace operations
        # NOTE: stack() increases in a 10% the cost in colab
        image_hls = torch.empty_like(image)
        h = torch.select(image_hls, -3, 0)
        l = torch.select(image_hls, -3, 1)
        s = torch.select(image_hls, -3, 2)
        torch.add(maxc, minc, out=l)  # l = max + min
        torch.sub(maxc, minc, out=s)  # s = max - min

    # precompute image / (max - min)
    im: torch.Tensor = image / (s + eps).unsqueeze(-3)

    # epsilon cannot be inside the torch.where to avoid precision issues
    s /= torch.where(l < 1.0, l, 2.0 - l) + eps  # saturation
    l /= 2  # luminance

    # note that r,g and b were previously div by (max - min)
    r: torch.Tensor = torch.select(im, -3, 0)
    g: torch.Tensor = torch.select(im, -3, 1)
    b: torch.Tensor = torch.select(im, -3, 2)
    # h[imax == 0] = (((g - b) / (max - min)) % 6)[imax == 0]
    # h[imax == 1] = (((b - r) / (max - min)) + 2)[imax == 1]
    # h[imax == 2] = (((r - g) / (max - min)) + 4)[imax == 2]
    cond: torch.Tensor = imax.unsqueeze(-3) == _RGB2HSL_IDX
    if image.requires_grad:
        h = torch.mul((g - b) % 6, torch.select(cond, -3, 0))
    else:
        torch.mul((g - b).remainder(6), torch.select(cond, -3, 0), out=h)
    h += torch.add(b - r, 2) * torch.select(cond, -3, 1)
    h += torch.add(r - g, 4) * torch.select(cond, -3, 2)
    # h = 2.0 * math.pi * (60.0 * h) / 360.0
    h *= math.pi / 3.0  # hue [0, 2*pi]

    if image.requires_grad:
        return torch.stack([h, l, s], dim=-3)
    return image_hls


def hls_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a HLS image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: HLS image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hls_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if not torch.jit.is_scripting():
        # weird way to use globals compiling with JIT even in the code not used by JIT...
        # __setattr__ can be removed if pytorch version is > 1.6.0 and then use:
        # hls_to_rgb.HLS2RGB = hls_to_rgb.HLS2RGB.to(image.device)
        hls_to_rgb.__setattr__('HLS2RGB', hls_to_rgb.HLS2RGB.to(image))  # type: ignore
        _HLS2RGB: torch.Tensor = hls_to_rgb.HLS2RGB  # type: ignore
    else:
        _HLS2RGB = torch.tensor([[[0.0]], [[8.0]], [[4.0]]], device=image.device, dtype=image.dtype)  # 3x1x1

    im: torch.Tensor = image.unsqueeze(-4)
    h: torch.Tensor = torch.select(im, -3, 0)
    l: torch.Tensor = torch.select(im, -3, 1)
    s: torch.Tensor = torch.select(im, -3, 2)
    h = h * (6 / math.pi)  # h * 360 / (2 * math.pi) / 30
    a = s * torch.min(l, 1.0 - l)

    # kr = (0 + h) % 12
    # kg = (8 + h) % 12
    # kb = (4 + h) % 12
    k: torch.Tensor = (h + _HLS2RGB) % 12

    # l - a * max(min(min(k - 3.0, 9.0 - k), 1), -1)
    mink = torch.min(k - 3.0, 9.0 - k)
    return torch.addcmul(l, a, mink.clamp_(min=-1.0, max=1.0), value=-1)


# tricks to speed up a little bit the conversions by presetting small tensors
# (in the functions they are moved to the proper device)
hls_to_rgb.__setattr__('HLS2RGB', torch.tensor([[[0.0]], [[8.0]], [[4.0]]]))  # 3x1x1
rgb_to_hls.__setattr__('RGB2HSL_IDX', torch.tensor([[[0.0]], [[1.0]], [[2.0]]]))  # 3x1x1


class RgbToHls(nn.Module):
    r"""Convert an image from RGB to HLS.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        HLS version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hls = RgbToHls()
        >>> output = hls(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_hls(image)


class HlsToRgb(nn.Module):
    r"""Convert an image from HLS to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - input: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Reference:
        https://en.wikipedia.org/wiki/HSL_and_HSV

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HlsToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return hls_to_rgb(image)
