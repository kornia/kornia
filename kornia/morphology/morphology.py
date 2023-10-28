from typing import List, Optional

import torch
import torch.nn.functional as F

__all__ = ["dilation", "erosion", "opening", "closing", "gradient", "top_hat", "bottom_hat"]


def _neight2channels_like_kernel(kernel: torch.Tensor) -> torch.Tensor:
    h, w = kernel.size()
    kernel = torch.eye(h * w, dtype=kernel.dtype, device=kernel.device)
    return kernel.view(h * w, 1, h, w)


def dilation(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "unfold",
) -> torch.Tensor:
    r"""Return the dilated image applying the same kernel in each channel.

    .. image:: _static/img/dilation.png

    The kernel must have 2 dimensions.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a non-flat
            structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: convolution is faster and less memory hungry, and unfold is more stable numerically

    Returns:
        Dilated image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> dilated_img = dilation(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == "geodesic":
        border_value = -max_val
        border_type = "constant"
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    if engine == "unfold":
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.max(output + neighborhood.flip((0, 1)), 4)
        output, _ = torch.max(output, 4)
    elif engine == "convolution":
        B, C, H, W = tensor.size()
        h_pad, w_pad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel)
        output, _ = F.conv2d(
            output.view(B * C, 1, h_pad, w_pad), reshape_kernel, padding=0, bias=neighborhood.view(-1).flip(0)
        ).max(dim=1)
        output = output.view(B, C, H, W)
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use 'convolution' or 'unfold'")
    return output.view_as(tensor)


def erosion(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "unfold",
) -> torch.Tensor:
    r"""Return the eroded image applying the same kernel in each channel.

    .. image:: _static/img/erosion.png

    The kernel must have 2 dimensions.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element (torch.Tensor, optional): Structuring element used for the grayscale dilation.
            It may be a non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if border_type is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: ``convolution`` is faster and less memory hungry, and ``unfold`` is more stable numerically

    Returns:
        Eroded image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(5, 5)
        >>> output = erosion(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == "geodesic":
        border_value = max_val
        border_type = "constant"
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    if engine == "unfold":
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.min(output - neighborhood, 4)
        output, _ = torch.min(output, 4)
    elif engine == "convolution":
        B, C, H, W = tensor.size()
        Hpad, Wpad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel)
        output, _ = F.conv2d(
            output.view(B * C, 1, Hpad, Wpad), reshape_kernel, padding=0, bias=-neighborhood.view(-1)
        ).min(dim=1)
        output = output.view(B, C, H, W)
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use 'convolution' or 'unfold'")

    return output


def opening(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "unfold",
) -> torch.Tensor:
    r"""Return the opened image, (that means, dilation after an erosion) applying the same kernel in each channel.

    .. image:: _static/img/opening.png

    The kernel must have 2 dimensions.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: convolution is faster and less memory hungry, and unfold is more stable numerically

    Returns:
       torch.Tensor: Opened image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> opened_img = opening(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    return dilation(
        erosion(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
            engine=engine,
        ),
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )


def closing(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "unfold",
) -> torch.Tensor:
    r"""Return the closed image, (that means, erosion after a dilation) applying the same kernel in each channel.

    .. image:: _static/img/closing.png

    The kernel must have 2 dimensions.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default is None and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: convolution is faster and less memory hungry, and unfold is more stable numerically

    Returns:
       Closed image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> closed_img = closing(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    return erosion(
        dilation(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
            engine=engine,
        ),
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )


# Morphological Gradient
def gradient(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "unfold",
) -> torch.Tensor:
    r"""Return the morphological gradient of an image.

    .. image:: _static/img/gradient.png

    That means, (dilation - erosion) applying the same kernel in each channel.
    The kernel must have 2 dimensions.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default is None and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: convolution is faster and less memory hungry, and unfold is more stable numerically

    Returns:
       Gradient image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> gradient_img = gradient(tensor, kernel)
    """

    return dilation(
        tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    ) - erosion(
        tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )


def top_hat(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "unfold",
) -> torch.Tensor:
    r"""Return the top hat transformation of an image.

    .. image:: _static/img/top_hat.png

    That means, (image - opened_image) applying the same kernel in each channel.
    The kernel must have 2 dimensions.

    See :func:`~kornia.morphology.opening` for details.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: convolution is faster and less memory hungry, and unfold is more stable numerically

    Returns:
       Top hat transformed image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> top_hat_img = top_hat(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    return tensor - opening(
        tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )


def bottom_hat(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "unfold",
) -> torch.Tensor:
    r"""Return the bottom hat transformation of an image.

    .. image:: _static/img/bottom_hat.png

    That means, (closed_image - image) applying the same kernel in each channel.
    The kernel must have 2 dimensions.

    See :func:`~kornia.morphology.closing` for details.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: convolution is faster and less memory hungry, and unfold is more stable numerically

    Returns:
       Top hat transformed image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> bottom_hat_img = bottom_hat(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    return (
        closing(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
            engine=engine,
        )
        - tensor
    )
