try:
    import kornia_rs
except ImportError:
    kornia_rs = None

import os
from enum import Enum

import torch
from torch.utils import dlpack  # TODO: remove this if kornia relies on torch>=1.10

from kornia.color import rgb_to_grayscale, rgba_to_rgb
from kornia.color.gray import grayscale_to_rgb
from kornia.color.rgb import rgb_to_rgba
from kornia.core import Device, Tensor
from kornia.core.check import KORNIA_CHECK


class ImageLoadType(Enum):
    r"""Enum to specify the desired image type."""
    UNCHANGED = 0
    GRAY8 = 1
    RGB8 = 2
    RGBA8 = 3
    GRAY32 = 4
    RGB32 = 5


def load_image_to_tensor(path_file: str, device: Device) -> Tensor:
    # load the file and decodes using kornia_rs. Internally it uses a package that
    # combines image-rs a self maintained version of the dlpack-rs. After the decoding,
    # the obtained stream bits are encapusalted to a cv::Tensor data structure without
    # memory ownership and passed as PyCapsule from rust to python.
    cv_tensor = kornia_rs.read_image_rs(path_file)
    # for convenience use the torch dlpack parser to get a zero copy torch.Tensor
    # TODO: evaluate other potential API so that we can return in numpy, jax, mxnet since
    # the kornia_rs cv::Tensor has this ability.
    th_tensor = dlpack.from_dlpack(cv_tensor)  # HxWx3
    # move the tensor to the desired device, move the data layout to CHW and clone
    # to return an owned data tensor.
    dev = device if isinstance(device, torch.device) or device is None else torch.device(device)
    return th_tensor.to(device=dev).permute(2, 0, 1).clone()  # CxHxW


def to_float32(image: Tensor) -> Tensor:
    KORNIA_CHECK(image.dtype == torch.uint8)
    return image.float() / 255.0


def to_uint8(image: Tensor) -> Tensor:
    KORNIA_CHECK(image.dtype == torch.float32)
    return image.mul(255.0).byte()


def load_image(path_file: str, desired_type: ImageLoadType, device: Device = "cpu") -> Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    Args:
        path_file: Path to a valid image file.
        desired_type: the desired image type, defined by color space and dtype.
        device: the device where you want to get your image placed.

    Return:
        Image tensor with shape :math:`(3,H,W)`.
    """
    if kornia_rs is None:
        raise ModuleNotFoundError("The io API is not available: `pip install kornia_rs` in a Linux system.")

    KORNIA_CHECK(os.path.isfile(path_file), f"Invalid file: {path_file}")
    image: Tensor = load_image_to_tensor(path_file, device)  # CxHxW

    if desired_type == ImageLoadType.UNCHANGED:
        return image
    elif desired_type == ImageLoadType.GRAY8:
        if image.shape[0] == 1 and image.dtype == torch.uint8:
            return image
        elif image.shape[0] == 3 and image.dtype == torch.uint8:
            gray8 = rgb_to_grayscale(image)
            return gray8
        elif image.shape[0] == 4 and image.dtype == torch.uint8:
            gray32 = rgb_to_grayscale(rgba_to_rgb(to_float32(image)))
            return to_uint8(gray32)
    elif desired_type == ImageLoadType.RGB8:
        if image.shape[0] == 3 and image.dtype == torch.uint8:
            return image
        elif image.shape[0] == 1 and image.dtype == torch.uint8:
            rgb8 = grayscale_to_rgb(image)
            return rgb8
    elif desired_type == ImageLoadType.RGBA8:
        if image.shape[0] == 3 and image.dtype == torch.uint8:
            rgba32 = rgb_to_rgba(to_float32(image), 0.0)
            return to_uint8(rgba32)
    elif desired_type == ImageLoadType.GRAY32:
        if image.shape[0] == 1 and image.dtype == torch.uint8:
            return to_float32(image)
        elif image.shape[0] == 3 and image.dtype == torch.uint8:
            gray32 = rgb_to_grayscale(to_float32(image))
            return gray32
        elif image.shape[0] == 4 and image.dtype == torch.uint8:
            gray32 = rgb_to_grayscale(rgba_to_rgb(to_float32(image)))
            return gray32
    elif desired_type == ImageLoadType.RGB32:
        if image.shape[0] == 3 and image.dtype == torch.uint8:
            return to_float32(image)
        elif image.shape[0] == 1 and image.dtype == torch.uint8:
            rgb32 = grayscale_to_rgb(to_float32(image))
            return rgb32
    else:
        raise NotImplementedError(f"Unknown type: {desired_type}")
    return Tensor([])


def write_image(path_file: str, image: Tensor) -> None:
    """Save an image file using the Kornia Rust backend.

    For now, we only support the writing of JPEG of the following types: RGB8.

    Args:
        path_file: Path to a valid image file.
        image: Image tensor with shape :math:`(3,H,W)`.

    Return:
        None.
    """
    if kornia_rs is None:
        raise ModuleNotFoundError("The io API is not available: `pip install kornia_rs` in a Linux system.")

    KORNIA_CHECK("jpg" in path_file[-3:], f"Invalid file extension: {path_file}")
    KORNIA_CHECK(image.dim() == 3 and image.shape[0] == 3, f"Invalid image shape: {image.shape}")
    KORNIA_CHECK(image.dtype == torch.uint8, f"Invalid image dtype: {image.dtype}")

    # create the image encoder
    image_encoder = kornia_rs.ImageEncoder()
    image_encoder.set_quality(100)

    # move the tensor to the cpu and clone to avoid memory ownership issues.
    image = image.cpu().clone()  # 3xHxW
    # move the data layout to HWC and convert to numpy
    image = image.permute(1, 2, 0).numpy()  # HxWx3

    # encode the image using the kornia_rs
    image_encoded: list = image_encoder.encode(image.tobytes(), image.shape)

    # save the image using the
    kornia_rs.write_image_jpeg(path_file, image_encoded)
