# based on: https://github.com/anguelos/tormentor/blob/e8050ac235b0c7ad3c7d931cfa47c308a305c486/diamond_square/diamond_square.py  # noqa: E501
import math
from typing import Callable, List, Optional, Tuple, Union

import torch

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.enhance import normalize_min_max
from kornia.filters import filter2d

# the default kernels for the diamond square
default_diamond_kernel: List[List[float]] = [[0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 0.25]]
default_square_kernel: List[List[float]] = [[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]


def _diamond_square_seed(
    replicates: int,
    width: int,
    height: int,
    random_fn: Callable[..., Tensor],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Helper function to generate the diamond square image seee.

    Args:
        replicates: the num of batched replicas for the image.
        width: the expected image width.
        height: the expected image height.
        random_fn: the random function to generate the image seed.
        device: the torch device where to create the image seed.
        dtype: the torch dtype where to create the image seed.

    Return:
        the generated image seed of size Bx1xHxW.
    """
    KORNIA_CHECK(width == 3 or height == 3, "Height or Width must be equal to 3.")
    # TODO(anguelos): can we avoid transposing and passing always fixed size. This will cause issues with onnx/jit
    transpose: bool = False
    if height == 3:
        transpose = True
        width, height = height, width

    # width is always 3
    KORNIA_CHECK(height % 2 == 1 and height > 2, "Height must be odd and height bigger than 2")

    res: Tensor = random_fn([replicates, 1, width, height], device=device, dtype=dtype)
    res[..., ::2, ::2] = random_fn([replicates, 1, 2, (height + 1) // 2], device=device, dtype=dtype)

    # Diamond step
    res[..., 1, 1::2] = (res[..., ::2, :-2:2] + res[..., ::2, 2::2]).sum(dim=2) / 4.0

    # Square step
    if width > 3:
        res[..., 1, 2:-3:2] = (
            res[..., 0, 2:-3:2] + res[..., 2, 2:-3:2] + res[..., 1, 0:-4:2] + res[..., 1, 2:-3:2]
        ) / 4.0

    tmp1 = res[..., 2, 0]
    res[..., 1, 0] = res[..., 0, 0] + res[..., 1, 1] + tmp1
    res[..., 1, -1] = res[..., -1, -1] + res[..., 1, -2] + tmp1
    tmp2 = res[..., 1, 1::2]
    res[..., 0, 1::2] = res[..., 0, 0:-2:2] + res[..., 0, 2::2] + tmp2
    res[..., 2, 1::2] = res[..., 2, 0:-2:2] + res[..., 2, 2::2] + tmp2
    res = res / 3.0

    if transpose:
        res = res.transpose(2, 3)
    return res


def _one_diamond_one_square(
    img: Tensor,
    random_scale: Union[float, Tensor],
    random_fn: Callable[..., Tensor] = torch.rand,
    diamond_kernel: Optional[Tensor] = None,
    square_kernel: Optional[Tensor] = None,
) -> Tensor:
    """Doubles the image resolution by applying a single diamond square steps.

    Recursive application of this method creates plasma fractals.

    Attention! The function is differentiable and gradients are computed as well.

    If this function is run in the usual sense, it is more efficient if it is run in a no_grad()

    Args:
        img: a 4D tensor where dimensions are Batch, Channel, Width, Height. Width and Height must both be 2^N+1 and
            Batch and Channels should in the usual case be 1.
        random_scale: a float  number in [0,1] controlling the randomness created pixels get. I the usual case, it is
            halved at every application of this function.
        random_fn: the random function to generate the image seed.
        diamond_kernel: the 3x3 kernel to perform the diamond step.
        square_kernel: the 3x3 kernel to perform the square step.

    Return:
        A tensor on the same device as img with the same channels as img and width, height of 2^(N+1)+1.
    """
    KORNIA_CHECK_SHAPE(img, ["B", "C", "H", "W"])
    # TODO (anguelos) test multi channel and batch size > 1

    if diamond_kernel is None:
        diamond_kernel = Tensor([default_diamond_kernel]).to(img)  # 1x3x3
    if square_kernel is None:
        square_kernel = Tensor([default_square_kernel]).to(img)  # 1x3x3

    batch_sz, _, height, width = img.shape
    new_img: Tensor = torch.zeros(
        [batch_sz, 1, 2 * (height - 1) + 1, 2 * (width - 1) + 1], device=img.device, dtype=img.dtype
    )
    new_img[:, :, ::2, ::2] = img

    factor: float = 1.0 / 0.75
    pad_compencate = torch.ones_like(new_img)
    pad_compencate[:, :, :, 0] = factor
    pad_compencate[:, :, :, -1] = factor
    pad_compencate[:, :, 0, :] = factor
    pad_compencate[:, :, -1, :] = factor

    random_img: Tensor = random_fn(new_img.size(), device=img.device, dtype=img.dtype) * random_scale

    # TODO(edgar): use kornia.filter2d
    # diamond
    diamond_regions = filter2d(new_img, diamond_kernel)
    diamond_centers = (diamond_regions > 0).to(img.dtype)
    # TODO (anguelos) make sure diamond_regions*diamond_centers is needed
    new_img = new_img + (1 - random_scale) * diamond_regions * diamond_centers + diamond_centers * random_img

    # square
    square_regions = filter2d(new_img, square_kernel) * pad_compencate
    square_centers = (square_regions > 0).to(img.dtype)

    # TODO (anguelos) make sure square_centers*square_regions is needed
    new_img = new_img + square_centers * random_img + (1 - random_scale) * square_centers * square_regions

    return new_img


def diamond_square(
    output_size: Tuple[int, int, int, int],
    roughness: Union[float, Tensor] = 0.5,
    random_scale: Union[float, Tensor] = 1.0,
    random_fn: Callable[..., Tensor] = torch.rand,
    normalize_range: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Generates Plasma Fractal Images using the diamond square algorithm.

    See: https://en.wikipedia.org/wiki/Diamond-square_algorithm

    Args:
        output_size: a tuple of integers with the BxCxHxW of the image to be generated.
        roughness: the scale value to apply at each recursion step.
        random_scale: the initial value of the scale for recursion.
        random_fn: the callable function to use to sample a random tensor.
        normalize_range: whether to normalize using min-max the output map. In case of a
            range is specified, min-max norm is applied between the provided range.
        device: the torch device to place the output map.
        dtype: the torch dtype to place the output map.

    Returns:
        A tensor with shape :math:`(B,C,H,W)` containing the fractal image.
    """
    KORNIA_CHECK(len(output_size) == 4, "output_size must be (B,C,H,W)")
    if not isinstance(random_scale, Tensor):
        random_scale = Tensor([[[[random_scale]]]]).to(device, dtype)
        random_scale = random_scale.expand([output_size[0] * output_size[1], 1, 1, 1])
    else:
        KORNIA_CHECK_IS_TENSOR(random_scale)
        random_scale = random_scale.view(-1, 1, 1, 1)
        random_scale = random_scale.expand([output_size[0], output_size[1], 1, 1])
        random_scale = random_scale.reshape([-1, 1, 1, 1])

    if not isinstance(roughness, Tensor):
        roughness = Tensor([[[[roughness]]]]).to(device, dtype)
        roughness = roughness.expand([output_size[0] * output_size[1], 1, 1, 1])
    else:
        roughness = roughness.view(-1, 1, 1, 1)
        roughness = roughness.expand([output_size[0], output_size[1], 1, 1])
        roughness = roughness.reshape([-1, 1, 1, 1])

    width, height = output_size[-2:]
    num_samples: int = 1
    for x in output_size[:-2]:
        num_samples *= x

    # compute the image seed
    p2_width: float = 2 ** math.ceil(math.log2(width - 1)) + 1
    p2_height: float = 2 ** math.ceil(math.log2(height - 1)) + 1
    recursion_depth: int = int(min(math.log2(p2_width - 1) - 1, math.log2(p2_height - 1) - 1))
    seed_width: int = (p2_width - 1) // 2**recursion_depth + 1
    seed_height: int = (p2_height - 1) // 2**recursion_depth + 1
    img: Tensor = random_scale * _diamond_square_seed(num_samples, seed_width, seed_height, random_fn, device, dtype)

    # perform recursion
    scale = random_scale
    for _ in range(recursion_depth):
        scale = scale * roughness
        img = _one_diamond_one_square(img, scale, random_fn)

    # slice to match with the output size
    img = img[..., :width, :height]
    img = img.view(output_size)

    # normalize the output in the range using min-max
    if normalize_range is not None:
        min_val, max_val = normalize_range
        img = normalize_min_max(img.contiguous(), min_val, max_val)
    return img
