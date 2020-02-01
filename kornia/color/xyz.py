from typing import Union

import torch
import torch.nn as nn



class RgbToXyz(nn.Module):
    r"""Convert image from RGB to XYZ
    The image data is assumed to be in the range of (0, 1).
    args:
        image (torch.Tensor): RGB image to be converted to XYZ.
    returns:
        torch.tensor: XYZ version of the image.
    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Examples::
        >>> img = torch.rand(2, 3, 4, 5)
        >>> xyz = kornia.color.RgbToXyz()
        >>> output = xyz(img)  
    Reference::
        [1] https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    """
    def __init__(self) -> None:
        super(RgbToXyz,self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_xyz(image)




class XyzToRgb(nn.Module):
    r"""Convert image from XYZ to RGB 
    args:
        image (torch.Tensor): XYZ image to be converted to RGB.
    returns:
        torch.tensor: RGB version of the image.
    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Examples::
        >>> img = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.XyzToRgb()
        >>> output = rgb(img)  
    Reference::
        [1] https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    """

    def __init__(self) -> None:
        super(XyzToRgb, self).__init__()
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return xyz_to_rgb(image)




def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to XYZ.

    See :class:`~kornia.color.RgbToXyz` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to XYZ.

    Returns:
        torch.Tensor: XYZ version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[...,0,:,:]
    g: torch.Tensor = image[...,1,:,:]
    b: torch.Tensor = image[...,2,:,:]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out:torch.Tensor = torch.stack((x,y,z), -3)

    return out




def xyz_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an XYZ image to RGB.

    See :class:`~kornia.color.XyzToRgb` for details.

    Args:
        image (torch.Tensor): XYZ Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.
    """    
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))
    
    x: torch.Tensor = image[...,0,:,:]
    y: torch.Tensor = image[...,1,:,:]
    z: torch.Tensor = image[...,2,:,:]

    r: torch.Tensor =  3.240479 * x + -1.537150 * y + -0.498535 * z
    g: torch.Tensor = -0.969256 * x +  1.875991 * y +  0.041556 * z
    b: torch.Tensor =  0.055648 * x + -0.204043 * y +  1.057311 * z

    out: torch.Tensor = torch.stack((r,b,g), dim=-3)

    return out