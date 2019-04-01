import torch
from torch.jit import ScriptModule, script_method


class Grayscale(ScriptModule):
    r"""Convert image to grayscale version of image.

    Uses torch.jit.ScriptModule to speed up operation.
    """
    def __init__(self) -> None:
        super(Grayscale, self).__init__()

    @script_method
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): Image to be converted to grayscale.

        Returns:
            torch.Tensor: Grayscale version of the image.
        """
        return rgb_to_grayscale(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def rgb_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to grayscale.

    Args:
        input (torch.Tensor): Image to be converted to grayscale.

    Returns:
        torch.Tensor: Grayscale version of the image.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if len(input.shape) < 3 and input.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
            .format(input.shape))

    # https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    gray: torch.Tensor = 0.299 * r + 0.587 * g + 0.110 * b
    return gray
