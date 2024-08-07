from kornia.core import Tensor, stack
from kornia.core.check import KORNIA_CHECK_IS_COLOR, KORNIA_CHECK_IS_TENSOR


def shift_rgb(image: Tensor, r_shift: Tensor, g_shift: Tensor, b_shift: Tensor) -> Tensor:
    """Shift rgb channels.

    Shift each image's channel by either r_shift for red, g_shift for green and b_shift for blue channels.
    """
    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK_IS_COLOR(image, f"with shape {image.shape}")

    shifts = [r_shift, g_shift, b_shift]

    shifted = (image + stack(shifts, dim=1).view(-1, 3, 1, 1).to(image)).clamp_(min=0, max=1)

    return shifted
