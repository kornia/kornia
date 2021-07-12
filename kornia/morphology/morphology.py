from typing import List, Optional

import torch
import torch.nn.functional as F


def dilation(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
) -> torch.Tensor:
    r"""Returns the dilated image applying the same kernel in each channel.

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

    Returns:
        Dilated image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> dilated_img = dilation(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(tensor.dim()))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Kernel type is not a torch.Tensor. Got {}".format(type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(kernel.dim()))

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == 'geodesic':
        border_value = -max_val
        border_type = 'constant'
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
    output, _ = torch.max(output + neighborhood.flip((0, 1)), 4)
    output, _ = torch.max(output, 4)

    return output


def erosion(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
) -> torch.Tensor:
    r"""Returns the eroded image applying the same kernel in each channel.

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

    Returns:
        Eroded image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(5, 5)
        >>> output = erosion(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(tensor.dim()))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Kernel type is not a torch.Tensor. Got {}".format(type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(kernel.dim()))

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == 'geodesic':
        border_value = max_val
        border_type = 'constant'
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
    output, _ = torch.min(output - neighborhood, 4)
    output, _ = torch.min(output, 4)

    return output


def opening(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
) -> torch.Tensor:
    r"""Returns the opened image, (that means, dilation after an erosion) applying the same kernel in each channel.

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

    Returns:
       torch.Tensor: Opened image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> opened_img = opening(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(tensor.dim()))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Kernel type is not a torch.Tensor. Got {}".format(type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(kernel.dim()))

    return dilation(
        erosion(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
        ),
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
    )


def closing(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
) -> torch.Tensor:
    r"""Returns the closed image, (that means, erosion after a dilation) applying the same kernel in each channel.

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

    Returns:
       Closed image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> closed_img = closing(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(tensor.dim()))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Kernel type is not a torch.Tensor. Got {}".format(type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(kernel.dim()))

    return erosion(
        dilation(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
        ),
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
    )


# Morphological Gradient
def gradient(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
) -> torch.Tensor:
    r"""Returns the morphological gradient of an image.

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

    Returns:
       Gradient image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

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
    ) - erosion(
        tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
    )


def top_hat(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
) -> torch.Tensor:
    r"""Returns the top hat tranformation of an image.

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

    Returns:
       Top hat transformated image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> top_hat_img = top_hat(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(tensor.dim()))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Kernel type is not a torch.Tensor. Got {}".format(type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(kernel.dim()))

    return tensor - opening(
        tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
    )


def bottom_hat(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
) -> torch.Tensor:
    r"""Returns the bottom hat tranformation of an image.

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

    Returns:
       Top hat transformated image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> bottom_hat_img = bottom_hat(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(tensor.dim()))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Kernel type is not a torch.Tensor. Got {}".format(type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(kernel.dim()))

    return (
        closing(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
        )
        - tensor
    )
