import keras_core as keras

from kornia.core import Module, IntegratedTensor


def _rgb_to_y(r: IntegratedTensor, g: IntegratedTensor, b: IntegratedTensor) -> IntegratedTensor:
    y: IntegratedTensor = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def rgb_to_ycbcr(image: IntegratedTensor) -> IntegratedTensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5
    """
    if not isinstance(image, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: IntegratedTensor = image[..., 0, :, :]
    g: IntegratedTensor = image[..., 1, :, :]
    b: IntegratedTensor = image[..., 2, :, :]

    delta: float = 0.5
    y: IntegratedTensor = _rgb_to_y(r, g, b)
    cb: IntegratedTensor = (b - y) * 0.564 + delta
    cr: IntegratedTensor = (r - y) * 0.713 + delta
    return keras.ops.stack([y, cb, cr], axis=-3)


def rgb_to_y(image: IntegratedTensor) -> IntegratedTensor:
    r"""Convert an RGB image to Y.

    Args:
        image: RGB Image to be converted to Y with shape :math:`(*, 3, H, W)`.

    Returns:
        Y version of the image with shape :math:`(*, 1, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_y(input)  # 2x1x4x5
    """
    if not isinstance(image, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: IntegratedTensor = image[..., 0:1, :, :]
    g: IntegratedTensor = image[..., 1:2, :, :]
    b: IntegratedTensor = image[..., 2:3, :, :]

    y: IntegratedTensor = _rgb_to_y(r, g, b)
    return y


def ycbcr_to_rgb(image: IntegratedTensor) -> IntegratedTensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: IntegratedTensor = image[..., 0, :, :]
    cb: IntegratedTensor = image[..., 1, :, :]
    cr: IntegratedTensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: IntegratedTensor = cb - delta
    cr_shifted: IntegratedTensor = cr - delta

    r: IntegratedTensor = y + 1.403 * cr_shifted
    g: IntegratedTensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: IntegratedTensor = y + 1.773 * cb_shifted
    return keras.ops.stack([r, g, b], axis=-3)


class RgbToYcbcr(IntegratedTensor): #CHECK nn.Module
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def call(self, image: IntegratedTensor) -> IntegratedTensor:
        return rgb_to_ycbcr(image)


class YcbcrToRgb(IntegratedTensor): #CHECK nn.Module
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def call(self, image: IntegratedTensor) -> IntegratedTensor:
        return ycbcr_to_rgb(image)
