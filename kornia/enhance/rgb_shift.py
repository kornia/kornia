import torch

from kornia.testing import KORNIA_CHECK_IS_COLOR, KORNIA_CHECK_IS_TENSOR


def shift_rgb(image, r_shift, g_shift, b_shift):
    """Shift rgb channels.

    Shift each image's channel by either r_shift for red, g_shift for green and b_shift for blue channels.
    """

    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK_IS_COLOR(image, f"with shape {image.shape}")

    shifts = [r_shift, g_shift, b_shift]

    shifted = (image + torch.Tensor(shifts).view(1, 3, 1, 1).to(image)).clamp_(min=0, max=1)

    return shifted
