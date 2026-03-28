# LICENSE HEADER MANAGED BY add-license-header
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = Canny()(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])


    """


    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False


    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        kernel_size: tuple[int, int] | int = (5, 5),
        sigma: tuple[float, float] | torch.Tensor = (1, 1),
        hysteresis: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """See :class:`Canny` for details."""
        super().__init__()


        KORNIA_CHECK(
            low_threshold <= high_threshold,
            "Invalid input thresholds. low_threshold should be smaller than the high_threshold. Got: "
            f"{low_threshold}>{high_threshold}",
        )
        KORNIA_CHECK(0 < low_threshold < 1, f"Invalid low threshold. Should be in range (0, 1). Got: {low_threshold}")
        KORNIA_CHECK(
            0 < high_threshold < 1, f"Invalid high threshold. Should be in range (0, 1). Got: {high_threshold}"
        )


        # Gaussian blur parameters
        self.kernel_size = kernel_size
        self.sigma = sigma


        # Double threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold


        # Hysteresis
        self.hysteresis = hysteresis


        self.eps: float = eps


    def __repr__(self) -> str:
        """See :class:`Canny` for details."""
        return "".join(
            (
                f"{type(self).__name__}(",
                ", ".join(
                    f"{name}={getattr(self, name)}" for name in sorted(self.__dict__) if not name.startswith("_")
                ),
                ")",
            )
        )


    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """See :class:`Canny` for details."""
        return canny(
            input, self.low_threshold, self.high_threshold, self.kernel_size, self.sigma, self.hysteresis, self.eps
        )

