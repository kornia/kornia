from typing import Tuple

import keras_core as keras

from kornia.core import Module, IntegratedTensor


def rgb_to_yuv(image: IntegratedTensor) -> IntegratedTensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: IntegratedTensor = image[..., 0, :, :]
    g: IntegratedTensor = image[..., 1, :, :]
    b: IntegratedTensor = image[..., 2, :, :]

    y: IntegratedTensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: IntegratedTensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: IntegratedTensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: IntegratedTensor = keras.ops.stack([y, u, v], axis=-3)

    return out

def _unfold2(input_tensor, dim, size, step):
    # Get the dimensions of the input tensor
    input_shape = keras.ops.shape(input_tensor)
    
    # Determine the number of slices
    num_slices = (input_shape[dim] - size) // step + 1
    
    # Initialize a list to store the unfolded slices
    unfolded_slices = []
    
    # Loop through the slices
    for i in range(num_slices):
        start = i * step
        end = start + size
        
        # Create a slice along the specified dimension
        if dim == -1:
            unfolded_slice = input_tensor[..., start:end]
        else:
            slices = [slice(None)] * len(input_shape)
            slices[dim] = slice(start, end)
            unfolded_slice = input_tensor[tuple(slices)]
        
        unfolded_slices.append(unfolded_slice)
    
    # Stack the slices along a new dimension
    unfolded_tensor = keras.ops.stack(unfolded_slices, axis=dim)
    
    return unfolded_tensor

def rgb_to_yuv420(image: IntegratedTensor) -> Tuple[IntegratedTensor, IntegratedTensor]:
    r"""Convert an RGB image to YUV 420 (subsampled).

    The image data is assumed to be in the range of (0, 1). Input need to be padded to be evenly divisible by 2
    horizontal and vertical. This function will output chroma siting (0.5,0.5)

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
        A Tensor containing the UV planes with shape :math:`(*, 2, H/2, W/2)`

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # (2x1x4x6, 2x2x2x3)
    """
    if not isinstance(image, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)
    yuvimage2 = _unfold2(yuvimage[..., 1:3, :, :],-2,2,2)
    yuvimage2 = _unfold2(yuvimage2,-2,2,2)
    yuvimage2 = keras.ops.mean(yuvimage2, axis=(-1, -2))
    return (yuvimage[..., :1, :, :], yuvimage2)


def rgb_to_yuv422(image: IntegratedTensor) -> Tuple[IntegratedTensor, IntegratedTensor]:
    r"""Convert an RGB image to YUV 422 (subsampled).

    The image data is assumed to be in the range of (0, 1). Input need to be padded to be evenly divisible by 2
    vertical. This function will output chroma siting (0.5)

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
       A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
       A Tensor containing the UV planes with shape :math:`(*, 2, H, W/2)`

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # (2x1x4x6, 2x1x4x3)
    """
    if not isinstance(image, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)
    yuvimage2 = _unfold2(yuvimage[..., 1:3, :, :],-1,2,2)
    yuvimage2 = keras.ops.mean(yuvimage2, axis=-1)

    return (yuvimage[..., :1, :, :], yuvimage2)


def yuv_to_rgb(image: IntegratedTensor) -> IntegratedTensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.

    Args:
        image: YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, IntegratedTensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: IntegratedTensor = image[..., 0, :, :]
    u: IntegratedTensor = image[..., 1, :, :]
    v: IntegratedTensor = image[..., 2, :, :]

    r: IntegratedTensor = y + 1.14 * v  # coefficient for g is 0
    g: IntegratedTensor = y + -0.396 * u - 0.581 * v
    b: IntegratedTensor = y + 2.029 * u  # coefficient for b is 0

    out: IntegratedTensor = keras.ops.stack([r, g, b], axis=-3)

    return out


def yuv420_to_rgb(imagey: IntegratedTensor, imageuv: IntegratedTensor) -> IntegratedTensor:
    r"""Convert an YUV420 image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.
    Input need to be padded to be evenly divisible by 2 horizontal and vertical.
    This function assumed chroma siting is (0.5, 0.5)

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        imageuv: UV (chroma) Image planes to be converted to RGB with shape :math:`(*, 2, H/2, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> output = yuv420_to_rgb(inputy, inputuv)  # 2x3x4x6
    """
    if not isinstance(imagey, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(imagey)}")

    if not isinstance(imageuv, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(imageuv)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imageuv.shape) < 3 or imageuv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of (*, 2, H/2, W/2). Got {imageuv.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if (
        len(imageuv.shape) < 2
        or len(imagey.shape) < 2
        or imagey.shape[-2] / imageuv.shape[-2] != 2
        or imagey.shape[-1] / imageuv.shape[-1] != 2
    ):
        raise ValueError(
            f"Input imageuv H&W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    imageuv2= keras.ops.repeat(imageuv,2, axis=-1)
    imageuv2= keras.ops.repeat(imageuv2,2, axis=-2)
    yuv444image = keras.ops.concatenate([imagey, imageuv2], axis=-3)
    # then convert the yuv444 tensor

    return yuv_to_rgb(yuv444image)


def yuv422_to_rgb(imagey: IntegratedTensor, imageuv: IntegratedTensor) -> IntegratedTensor:
    r"""Convert an YUV422 image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.
    Input need to be padded to be evenly divisible by 2 vertical. This function assumed chroma siting is (0.5)

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        imageuv: UV (luma) Image planes to be converted to RGB with shape :math:`(*, 2, H, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> output = yuv420_to_rgb(inputy, inputuv)  # 2x3x4x5
    """
    if not isinstance(imagey, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(imagey)}")

    if not isinstance(imageuv, IntegratedTensor):
        raise TypeError(f"Input type is not a IntegratedTensor. Got {type(imageuv)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imageuv.shape) < 3 or imageuv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of (*, 2, H, W/2). Got {imageuv.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if len(imageuv.shape) < 2 or len(imagey.shape) < 2 or imagey.shape[-1] / imageuv.shape[-1] != 2:
        raise ValueError(
            f"Input imageuv W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = keras.ops.concatenate([imagey, keras.ops.repeat(imageuv,2, axis=-1)], axis=-3)
    # then convert the yuv444 tensor
    return yuv_to_rgb(yuv444image)


class RgbToYuv(IntegratedTensor): #CHECK nn.Module
    r"""Convert an image from RGB to YUV.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YUV version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> yuv = RgbToYuv()
        >>> output = yuv(input)  # 2x3x4x5

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def call(self, input: IntegratedTensor) -> IntegratedTensor:
        return rgb_to_yuv(input)


class RgbToYuv420(IntegratedTensor): #CHECK nn.Module
    r"""Convert an image from RGB to YUV420.

    The image data is assumed to be in the range of (0, 1). Width and Height evenly divisible by 2.

    Returns:
        YUV420 version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)` and :math:`(*, 2, H/2, W/2)`

    Examples:
        >>> yuvinput = torch.rand(2, 3, 4, 6)
        >>> yuv = RgbToYuv420()
        >>> output = yuv(yuvinput)  # # (2x1x4x6, 2x1x2x3)

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def call(self, yuvinput: IntegratedTensor) -> Tuple[IntegratedTensor, IntegratedTensor]:  # skipcq: PYL-R0201
        return rgb_to_yuv420(yuvinput)


class RgbToYuv422(IntegratedTensor): #CHECK nn.Module
    r"""Convert an image from RGB to YUV422.

    The image data is assumed to be in the range of (0, 1). Width evenly disvisible by 2.

    Returns:
        YUV422 version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)` and :math:`(*, 2, H, W/2)`

    Examples:
        >>> yuvinput = torch.rand(2, 3, 4, 6)
        >>> yuv = RgbToYuv422()
        >>> output = yuv(yuvinput)  # # (2x1x4x6, 2x2x4x3)

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def call(self, yuvinput: IntegratedTensor) -> Tuple[IntegratedTensor, IntegratedTensor]:  # skipcq: PYL-R0201
        return rgb_to_yuv422(yuvinput)


class YuvToRgb(IntegratedTensor): #CHECK nn.Module
    r"""Convert an image from YUV to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def call(self, input: IntegratedTensor) -> IntegratedTensor:
        return yuv_to_rgb(input)


class Yuv420ToRgb(IntegratedTensor): #CHECK nn.Module
    r"""Convert an image from YUV to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.
    Width and Height evenly divisible by 2.

    Returns:
        RGB version of the image.

    Shape:
        - imagey: :math:`(*, 1, H, W)`
        - imageuv: :math:`(*, 2, H/2, W/2)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> rgb = Yuv420ToRgb()
        >>> output = rgb(inputy, inputuv)  # 2x3x4x6
    """

    def call(self, inputy: IntegratedTensor, inputuv: IntegratedTensor) -> IntegratedTensor:  # skipcq: PYL-R0201
        return yuv420_to_rgb(inputy, inputuv)


class Yuv422ToRgb(IntegratedTensor): #CHECK nn.Module
    r"""Convert an image from YUV to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.
    Width evenly divisible by 2.

    Returns:
        RGB version of the image.

    Shape:
        - imagey: :math:`(*, 1, H, W)`
        - imageuv: :math:`(*, 2, H, W/2)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 4, 3)
        >>> rgb = Yuv422ToRgb()
        >>> output = rgb(inputy, inputuv)  # 2x3x4x6
    """

    def call(self, inputy: IntegratedTensor, inputuv: IntegratedTensor) -> IntegratedTensor:  # skipcq: PYL-R0201
        return yuv422_to_rgb(inputy, inputuv)
