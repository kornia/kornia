# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# based on: https://github.com/anguelos/tormentor/blob/e8050ac235b0c7ad3c7d931cfa47c308a305c486/diamond_square/diamond_square.py
import math
from typing import Callable, List, Optional, Tuple, Union

import torch

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
    random_fn: Callable[..., torch.Tensor],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate the diamond square image seee.

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

    # `width` is always 3 here: rows 0 and 2 hold the corner draws, row 1 the midpoints. `height` is the odd length K.
    res = torch.zeros([replicates, 1, width, height], device=device, dtype=dtype)
    res[..., ::2, ::2] = random_fn([replicates, 1, 2, (height + 1) // 2], device=device, dtype=dtype)

    # diamond step: the centre of each cell is the mean of its four corners
    res[..., 1, 1::2] = (res[..., 0, :-2:2] + res[..., 0, 2::2] + res[..., 2, :-2:2] + res[..., 2, 2::2]) / 4.0

    # square step: a border point has three neighbours, an interior one four
    res[..., 0, 1::2] = (res[..., 0, :-2:2] + res[..., 0, 2::2] + res[..., 1, 1::2]) / 3.0
    res[..., 2, 1::2] = (res[..., 2, :-2:2] + res[..., 2, 2::2] + res[..., 1, 1::2]) / 3.0
    res[..., 1, 0] = (res[..., 0, 0] + res[..., 2, 0] + res[..., 1, 1]) / 3.0
    res[..., 1, -1] = (res[..., 0, -1] + res[..., 2, -1] + res[..., 1, -2]) / 3.0
    if height > 3:
        res[..., 1, 2:-1:2] = (
            res[..., 0, 2:-1:2] + res[..., 2, 2:-1:2] + res[..., 1, 1:-2:2] + res[..., 1, 3::2]
        ) / 4.0

    if transpose:
        res = res.transpose(2, 3)
    return res


def _one_diamond_one_square(
    img: torch.Tensor,
    random_scale: Union[float, torch.Tensor],
    random_fn: Callable[..., torch.Tensor] = torch.rand,
    diamond_kernel: Optional[torch.Tensor] = None,
    square_kernel: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Doubles the image resolution by applying a single diamond square steps.

    Recursive application of this method creates plasma fractals.

    Attention! The function is differentiable and gradients are computed as well.

    If this function is run in the usual sense, it is more efficient if it is run in a no_grad()

    Args:
        img: a 4D torch.Tensor where dimensions are Batch, Channel, Width, Height.
            Width and Height must both be 2^N+1 and
            Batch and Channels should in the usual case be 1.
        random_scale: a float  number in [0,1] controlling the randomness created pixels get. I the usual case, it is
            halved at every application of this function.
        random_fn: the random function to generate the image seed.
        diamond_kernel: the 3x3 kernel to perform the diamond step.
        square_kernel: the 3x3 kernel to perform the square step.

    Return:
        A torch.Tensor on the same device as img with the same channels as img and width, height of 2^(N+1)+1.

    """
    KORNIA_CHECK_SHAPE(img, ["B", "C", "H", "W"])
    # TODO (anguelos) test multi channel and batch size > 1

    if diamond_kernel is None:
        diamond_kernel = torch.Tensor([default_diamond_kernel]).to(img)  # 1x3x3
    if square_kernel is None:
        square_kernel = torch.Tensor([default_square_kernel]).to(img)  # 1x3x3

    batch_sz, _, height, width = img.shape
    new_img: torch.Tensor = torch.zeros(
        [batch_sz, 1, 2 * (height - 1) + 1, 2 * (width - 1) + 1], device=img.device, dtype=img.dtype
    )
    new_img[:, :, ::2, ::2] = img

    factor: float = 1.0 / 0.75
    pad_compencate = torch.ones_like(new_img)
    pad_compencate[:, :, :, 0] = factor
    pad_compencate[:, :, :, -1] = factor
    pad_compencate[:, :, 0, :] = factor
    pad_compencate[:, :, -1, :] = factor

    random_img: torch.Tensor = random_fn(new_img.size(), device=img.device, dtype=img.dtype) * random_scale

    # the diamond and square centres are fixed by index parity, not by the sign of the filtered values
    rows = torch.arange(new_img.shape[-2], device=img.device).view(-1, 1) % 2
    cols = torch.arange(new_img.shape[-1], device=img.device).view(1, -1) % 2
    diamond_centers = ((rows == 1) & (cols == 1)).to(img.dtype)
    square_centers = ((rows + cols) == 1).to(img.dtype)

    # diamond
    diamond_regions = filter2d(new_img, diamond_kernel)
    new_img = new_img + (1 - random_scale) * diamond_regions * diamond_centers + diamond_centers * random_img

    # square: zero padding + 4/3 turns a border point's three-neighbour sum into their mean
    # ('reflect' would duplicate a neighbour)
    square_regions = filter2d(new_img, square_kernel, border_type="constant") * pad_compencate
    return new_img + square_centers * random_img + (1 - random_scale) * square_centers * square_regions


def diamond_square(
    output_size: Tuple[int, int, int, int],
    roughness: Union[float, torch.Tensor] = 0.5,
    random_scale: Union[float, torch.Tensor] = 1.0,
    random_fn: Callable[..., torch.Tensor] = torch.rand,
    normalize_range: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Generate Plasma Fractal Images using a variant of the diamond square algorithm.

    It ports `tormentor <https://github.com/anguelos/tormentor>`_'s variant of the
    `diamond-square algorithm <https://en.wikipedia.org/wiki/Diamond-square_algorithm>`_.

    Convention:
        - The map is grown on a :math:`2^n + 1` grid and cropped to ``output_size`` at its top-left corner, so it can
          be a non-contiguous view. The seed is a :math:`3 \times K` grid, transposed to :math:`K \times 3` for a map
          taller than wide, whose cell corners, the even nodes of its first and last rows (columns when transposed),
          are ``random_fn`` draws, and whose other nodes are the means of their neighbours.
          Each level sets a new node to
          :math:`(1 - s_k)\,\overline{n} + s_k\,u`, the mean :math:`\overline{n}` of its neighbours mixed with a
          ``random_fn`` draw :math:`u` by the weight :math:`s_k = \mathrm{random\_scale} \cdot \mathrm{roughness}^k`
          at level :math:`k \geq 1`. ``random_scale`` and ``roughness`` act only through :math:`s_k`, a mixing
          weight, not an amplitude. A border node of the square step averages its three neighbours; the diamond
          step's nodes are all interior, so its ``reflect`` padding never reaches them.
        - ``random_fn`` is called as ``random_fn(shape, device=device, dtype=dtype)`` and is the only source of
          randomness, so ``functools.partial(torch.rand, generator=g)`` leaves the global generator alone. With
          ``torch.rand`` the map stays in [0, 1] without normalization while ``random_scale`` and ``roughness`` are in
          [0, 1], and it is flat in mean (0.5 at every pixel) for every ``random_scale``, though its variance varies
          across the map.
        - ``roughness`` and ``random_scale`` are a float, a 0-d tensor or a ``(B,)`` tensor with one value per
          sample, and follow ``device`` and ``dtype``; with ``dtype=None`` a tensor keeps its dtype and a float takes
          the default dtype. When ``random_fn`` returns the dtype it is asked for, the map has the dtype that
          ``dtype`` (the default dtype for ``None``), ``roughness`` and ``random_scale`` promote to, at every map
          size. A ``random_fn`` that returns another dtype also promotes the levels' arithmetic to its dtype, but not
          a map with a side of 3 px or less, which is the seed alone and stores the draws in the requested dtype.
        - ``normalize_range=(min, max)`` rescales each :math:`(b, c)` plane to that range with
          :func:`~kornia.enhance.normalize_min_max`.

    Args:
        output_size: a tuple of integers with the BxCxHxW of the image to be generated.
        roughness: the factor that scales the mixing weight at each recursion step.
        random_scale: scales the per-level mixing weight
            :math:`s_k = \mathrm{random\_scale} \cdot \mathrm{roughness}^k`, :math:`k \geq 1`.
        random_fn: the callable function to use to sample a random torch.Tensor.
        normalize_range: if given, the ``(min, max)`` range each plane is min-max normalized to.
        device: the torch device to place the output map.
        dtype: the torch dtype to place the output map.

    Returns:
        A torch.Tensor with shape ``(B,C,H,W)`` containing the fractal image.

    """
    KORNIA_CHECK(len(output_size) == 4, "output_size must be (B,C,H,W)")
    if not isinstance(random_scale, torch.Tensor):
        random_scale = torch.Tensor([[[[random_scale]]]]).to(device, dtype)
        random_scale = random_scale.expand([output_size[0] * output_size[1], 1, 1, 1])
    else:
        KORNIA_CHECK_IS_TENSOR(random_scale)
        random_scale = random_scale.to(device=device, dtype=dtype).view(-1, 1, 1, 1)
        random_scale = random_scale.expand([output_size[0], output_size[1], 1, 1])
        random_scale = random_scale.reshape([-1, 1, 1, 1])

    if not isinstance(roughness, torch.Tensor):
        roughness = torch.Tensor([[[[roughness]]]]).to(device, dtype)
        roughness = roughness.expand([output_size[0] * output_size[1], 1, 1, 1])
    else:
        roughness = roughness.to(device=device, dtype=dtype).view(-1, 1, 1, 1)
        roughness = roughness.expand([output_size[0], output_size[1], 1, 1])
        roughness = roughness.reshape([-1, 1, 1, 1])

    width, height = output_size[-2:]
    num_samples: int = 1
    for x in output_size[:-2]:
        num_samples *= x

    # compute the image seed. The smallest grid the algorithm builds is 3 px per side (two corners
    # and a midpoint), so a smaller side is generated at 3 px and sliced down below: a 1 or 2 px
    # side is just its corner draws. Sizes from 3 px up are unaffected.
    p2_width: int = 2 ** math.ceil(math.log2(max(width, 3) - 1)) + 1
    p2_height: int = 2 ** math.ceil(math.log2(max(height, 3) - 1)) + 1
    recursion_depth: int = int(min(math.log2(p2_width - 1) - 1, math.log2(p2_height - 1) - 1))
    seed_width: int = (p2_width - 1) // 2**recursion_depth + 1
    seed_height: int = (p2_height - 1) // 2**recursion_depth + 1
    seed = _diamond_square_seed(num_samples, seed_width, seed_height, random_fn, device, dtype)
    # The output dtype is that of the levels' arithmetic on the actual draws. Level 1 asks random_fn for the seed's
    # dtype promoted with random_scale's, and each level promotes the map with its draws and with
    # random_scale * roughness**k. A map that is the seed alone has no level, so it is promoted with roughness's dtype
    # after the loop; past the seed that promotion is already done and the cast is a no-op.
    img: torch.Tensor = seed.to(torch.promote_types(seed.dtype, random_scale.dtype))

    # perform recursion
    scale = random_scale
    for _ in range(recursion_depth):
        scale = scale * roughness
        img = _one_diamond_one_square(img, scale, random_fn)
    img = img.to(torch.promote_types(img.dtype, roughness.dtype))

    # slice to match with the output size
    img = img[..., :width, :height]
    img = img.view(output_size)

    # min-max normalize each (b, c) plane to the range
    if normalize_range is not None:
        min_val, max_val = float(normalize_range[0]), float(normalize_range[1])
        img = normalize_min_max(img.contiguous(), min_val, max_val)
    return img
