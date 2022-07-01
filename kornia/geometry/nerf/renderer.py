import torch


class VolumeRenderer(torch.nn.Module):
    _huge = 1.0e10
    _eps = 1.0e-10

    def __init__(self, shift: int = 1) -> None:
        super().__init__()
        self._shift = shift

    def _render(self, alpha: torch.Tensor, rgbs: torch.Tensor) -> torch.Tensor:
        trans = torch.cumprod(1 - alpha + self._eps, dim=-1)  # (*, N)
        trans = torch.roll(trans, shifts=self._shift, dims=-1)  # (*, N)
        trans[..., : self._shift] = 1  # (*, N)

        weights = trans * alpha  # (*, N)

        rgbs_rendered = torch.sum(weights[..., None] * rgbs, dim=-2)  # (*, 3)

        return rgbs_rendered


class IrregularRenderer(VolumeRenderer):
    def __init__(self, shift: int = 1) -> None:
        super().__init__(shift)

    def forward(self, rgbs: torch.Tensor, densities: torch.Tensor, t_vals: torch.Tensor) -> torch.Tensor:
        r"""Renders 3D irregularly sampled points along rays.       # FIXME: Add Mildenhall (2020) as a reference

        Args:
            rgbs: RGB values of points along rays :math:`(*, N, 3)`
            densities: Volume densities of points along rays :math:`(*, N)`
            t_vals: Sampled point distances along rays :math: `(*, N)`

        Returns:
            Rendered RGB values for each ray :math:`(*, 3)`

        Examples:       # FIXME: Fix this example!!
            >>> input = torch.tensor([[0., 0.]])
            >>> convert_points_to_homogeneous(input)
            tensor([[0., 0., 1.]])
        """
        deltas = t_vals[..., 1:] - t_vals[..., :-1]  # (*, N - 1)
        far = torch.empty(size=t_vals.shape[:-1], dtype=t_vals.dtype, device=t_vals.device).fill_(self._huge)
        deltas = torch.cat([deltas, far[..., None]], dim=-1)  # (*, N)

        alpha = 1 - torch.exp(-1.0 * densities * deltas)  # (*, N)

        return self._render(alpha, rgbs)


class RegularRenderer(VolumeRenderer):
    def __init__(self, shift: int = 1) -> None:
        super().__init__(shift)

    def forward(self, rgbs: torch.Tensor, densities: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        r"""Renders 3D regularly sampled points along rays.       # FIXME: Add Mildenhall (2020) as a reference

        Args:
            rgbs: RGB values of points along rays :math:`(*, N, 3)`
            densities: Volume densities of points along rays :math:`(*, N)`
            deltas: Equispaced distance for point pairs along each ray :math: `(*)`

        Returns:
            Rendered RGB values for each ray :math:`(*, 3)`

        Examples:       # FIXME: Fix this example!!
            >>> input = torch.tensor([[0., 0.]])
            >>> convert_points_to_homogeneous(input)
            tensor([[0., 0., 1.]])
        """
        alpha = 1 - torch.exp(-1.0 * densities * deltas[..., None])  # (*, N)
        alpha[..., -1] = 0

        return self._render(alpha, rgbs)
