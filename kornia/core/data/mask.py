from .image import Image


class Mask(Image):  # B, N, C, H, W
    """Mask is a special image format with pixel value restrictions.
    """