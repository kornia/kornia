import torch


def shift_rgb(image, r_shift, g_shift, b_shift):
    """
    Shift each image's channel by either r_shift for red, g_shift for green and b_shift for blue channels.
    """
    if r_shift == g_shift == b_shift:
        return image + r_shift

    shifts = [r_shift, g_shift, b_shift]

    shifted = (image + torch.Tensor(shifts).view(1, 3, 1, 1).to(image)).clamp_(min=0, max=1)

    return shifted
