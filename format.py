class Dummy:
    def _compute_affine_matrix(
        angle: Optional[torch.Tensor],
        translation: Optional[torch.Tensor],
        scale_factor: Optional[torch.Tensor],
        shear: Optional[torch.Tensor],
        center: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass

    def __init__(
        self,
        angle: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        scale_factor: Optional[torch.Tensor] = None,
        shear: Optional[torch.Tensor] = None,
        center: Optional[torch.Tensor] = None,
        align_corners: bool = False,
    ) -> None:
        pass
