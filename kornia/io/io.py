from __future__ import annotations

try:
    import kornia_rs
except ImportError:  # pragma: no cover
    kornia_rs = None

from enum import Enum
from pathlib import Path

import torch

from kornia.color import rgb_to_grayscale, rgba_to_rgb
from kornia.color.gray import grayscale_to_rgb
from kornia.color.rgb import rgb_to_rgba
from kornia.core import Device, Tensor
from kornia.core.check import KORNIA_CHECK
from kornia.utils import image_to_tensor, tensor_to_image


class ImageLoadType(Enum):
    r"""Enum to specify the desired image type."""

    UNCHANGED = 0
    GRAY8 = 1
    RGB8 = 2
    RGBA8 = 3
    GRAY32 = 4
    RGB32 = 5


def _load_image_to_tensor(path_file: Path, device: Device) -> Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    The decoded image is returned as numpy array with shape HxWxC.

    Args:
        path_file: Path to a valid image file.
        device: the device where you want to get your image placed.

    Return:
        Image tensor with shape :math:`(3,H,W)`.
    """

    # read image and return as `np.ndarray` with shape HxWxC
    if path_file.suffix in [".jpg", ".jpeg"]:
        img = kornia_rs.read_image_jpeg(str(path_file))
    else:
        img = kornia_rs.read_image_any(str(path_file))

    # convert the image to tensor with shape CxHxW
    img_t = image_to_tensor(img, keepdim=True)

    # move the tensor to the desired device,
    dev = device if isinstance(device, torch.device) or device is None else torch.device(device)

    return img_t.to(device=dev)


def to_float32(image: Tensor) -> Tensor:
    """Convert an image tensor to float32."""
    KORNIA_CHECK(image.dtype == torch.uint8)
    return image.float() / 255.0


def to_uint8(image: Tensor) -> Tensor:
    """Convert an image tensor to uint8."""
    KORNIA_CHECK(image.dtype == torch.float32)
    return image.mul(255.0).byte()


def load_image(path_file: str | Path, desired_type: ImageLoadType, device: Device = "cpu") -> Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    Args:
        path_file: Path to a valid image file.
        desired_type: the desired image type, defined by color space and dtype.
        device: the device where you want to get your image placed.

    Return:
        Image tensor with shape :math:`(3,H,W)`.
    """
    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    # read the image using the kornia_rs package
    image: Tensor = _load_image_to_tensor(path_file, device)  # CxHxW

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

    raise NotImplementedError(f"Unknown type: {desired_type}")


def write_image(path_file: str | Path, image: Tensor) -> None:
    """Save an image file using the Kornia Rust backend.

    For now, we only support the writing of JPEG of the following types: RGB8.

    Args:
        path_file: Path to a valid image file.
        image: Image tensor with shape :math:`(3,H,W)`.

    Return:
        None.
    """
    if kornia_rs is None:  # pragma: no cover
        raise ModuleNotFoundError("The io API is not available: `pip install kornia_rs` in a Linux system.")

    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    KORNIA_CHECK(path_file.suffix in [".jpg", ".jpeg"], f"Invalid file extension: {path_file}")
    KORNIA_CHECK(image.dim() == 3 and image.shape[0] == 3, f"Invalid image shape: {image.shape}")
    KORNIA_CHECK(image.dtype == torch.uint8, f"Invalid image dtype: {image.dtype}")

    # create the image encoder
    # image_encoder = kornia_rs.ImageEncoder()
    # image_encoder.set_quality(100)

    ## move the tensor to the cpu and clone to avoid memory ownership issues.
    # image = image.cpu().clone()  # 3xHxW

    ## move the data layout to HWC and convert to numpy
    # image_np = image.permute(1, 2, 0).numpy()  # HxWx3

    ## encode the image using the kornia_rs
    # image_encoded: list[int] = image_encoder.encode(image_np.tobytes(), image_np.shape)

    img_np = tensor_to_image(image, keepdim=True, force_contiguous=True)  # HxWx3

    # save the image using the
    kornia_rs.write_image_jpeg(str(path_file), img_np)
