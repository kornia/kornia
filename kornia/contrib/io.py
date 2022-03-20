try:
    import kornia_rs
except ImportError:
    kornia_rs = None

import torch

from kornia.core import Tensor


def read_image(path_file: str) -> Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    Args:
        path_file: Path to a valid image file.

    Return:
        Returns an image tensor with shape :math:`(3,H,W)`.
    """
    if kornia_rs is None:
        raise ModuleNotFoundError("The io API is not available: `pip install kornia_rs` in a Linux system.")

    cv_tensor = kornia_rs.read_image_rs(path_file)
    th_tensor = torch.utils.dlpack.from_dlpack(cv_tensor)  # HxWx3
    return th_tensor.permute(2, 0, 1)  # CxHxW
