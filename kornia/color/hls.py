import torch
import torch.nn as nn
import cv2
import kornia


class HlsToRgb(nn.Module):
    r"""Convert image from HLS to Rgb
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): HLS image to be converted to RGB.

    returns:
        torch.tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self) -> None:
        super(HlsToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return hls_to_rgb(image)


#def hls_to_rgb(image):







class RgbToHls(nn.Module):
    r"""Convert image from RGB to HLS
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to HLS.

    returns:
        torch.tensor: HLS version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self) -> None:
        super(RgbToHls, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_hls(image)



def rgb_to_hls(image):
    r"""Convert an RGB image to HLS
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): RGB Image to be converted to HLS.


    Returns:
        torch.Tensor: HLS version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]

    l: torch.Tensor = (maxc+minc)/2 # luminance

    deltac: torch.Tensor = maxc - minc
    print(deltac.dtype)
    s: torch.Tensor = torch.zeros((image.shape[0],image.shape[2],image.shape[3]),dtype=torch.float32)
    s[l<0.5]: torch.Tensor = (deltac[l<0.5]) / (maxc[l<0.5] + minc[l<0.5])
    s[l>=0.5]: torch.Tensor = (deltac[l>=0.5]) / (2 - (maxc[l>=0.5] + minc[l>=0.5]))

    rc: torch.Tensor = (maxc - r) / deltac
    gc: torch.Tensor = (maxc - g) / deltac
    bc: torch.Tensor = (maxc - b) / deltac

    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc

    h: torch.Tensor = 4.0 + gc - rc
    h[maxg]: torch.Tensor = 2.0 + rc[maxg] - bc[maxg]
    h[maxr]: torch.Tensor = bc[maxr] - gc[maxr]
    h[minc == maxc]: torch.Tensor = 0.0

    h: torch.Tensor = (h / 6.0) % 1.0

    return torch.stack([h, l, s], dim=-3)

#img = cv2.imread("../../../../Coloredhouses.jpg")
#img_t = kornia.image_to_tensor(img)
#img_rgb = kornia.bgr_to_rgb(img_t)
#img_hls = rgb_to_hls(img_rgb.float()/255)
