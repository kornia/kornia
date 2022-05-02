import torch


def shift_rgb(image, r_shift, g_shift, b_shift):
    """
    Shift each image's channel by either r_shift for red, g_shift for green and b_shift for blue channels.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    shifts = [r_shift, g_shift, b_shift]

    shifted = (image + torch.Tensor(shifts).view(1, 3, 1, 1).to(image)).clamp_(min=0, max=1)

    return shifted
