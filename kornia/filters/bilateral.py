# LICENSE HEADER MANAGED BY add-license-header
    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> blur = BilateralBlur((3, 3), 0.1, (1.5, 1.5))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])


    """


    def forward(self, input: torch.Tensor) -> torch.Tensor:
                """See :class:`BilateralBlur` for details."""
        return bilateral_blur(
            input, self.kernel_size, self.sigma_color, self.sigma_space, self.border_type, self.color_distance_type
        )




class JointBilateralBlur(_BilateralBlur):
    r"""Blur a torch.Tensor using a Joint Bilateral filter.


    This operator is almost identical to a Bilateral filter. The only difference
    is that the color Gaussian kernel is computed based on another image called
    a guidance image. See :class:`BilateralBlur` for more information.


    Arguments:
        kernel_size: the size of the kernel.
        sigma_color: the standard deviation for intensity/color Gaussian kernel.
          Smaller values preserve more edges.
        sigma_space: the standard deviation for spatial Gaussian kernel.
          This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        color_distance_type: the type of distance to calculate intensity/color
          difference. Only ``'l1'`` or ``'l2'`` is allowed. Use ``'l1'`` to
          match OpenCV implementation.


    Returns:
        the blurred input torch.Tensor.


    Shape:
        - Input: :math:`(B, C, H, W)`, :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`


    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> guidance = torch.rand(2, 4, 5, 5)
        >>> blur = JointBilateralBlur((3, 3), 0.1, (1.5, 1.5))
        >>> output = blur(input, guidance)
        >>> output.shape
        torch.Size([2, 4, 5, 5])


    """


    def forward(self, input: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        return joint_bilateral_blur(
            input,
            guidance,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )

