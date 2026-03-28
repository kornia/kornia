











    Returns:
        the blurred input torch.Tensor.
















    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`
















    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])
















    """
















    def __init__(
        self, kernel_size: tuple[int, int] | int, border_type: str = "reflect", separable: bool = False
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.separable = separable
















        if separable:
            ky, kx = _unpack_2d_ks(self.kernel_size)
            self.register_buffer("kernel_y", get_box_kernel1d(ky))
            self.register_buffer("kernel_x", get_box_kernel1d(kx))
            self.kernel_y: torch.Tensor
            self.kernel_x: torch.Tensor
        else:
            self.register_buffer("kernel", get_box_kernel2d(kernel_size))
            self.kernel: torch.Tensor
















    def __repr__(self) -> str:
        """See :class:`BoxBlur` for details."""
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"border_type={self.border_type}, "
            f"separable={self.separable})"
        )
















    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """See :class:`BoxBlur` for details."""
        KORNIA_CHECK_IS_TENSOR(input)
        if self.separable:
            return filter2d_separable(input, self.kernel_x, self.kernel_y, self.border_type)
        return filter2d(input, self.kernel, self.border_type)















