import torch

from kornia.core import Module, Tensor


class DiscreteSteerer(Module):
    """Module for discrete rotation steerers.

    A steerer rotates keypoint descriptions in latent space as if they were obtained from rotated images.

    Args:
        generator: [N, N] tensor where N is the descriptor dimension.

    Example:
        >>> desc = torch.randn(512, 128)
        >>> generator = torch.randn(128, 128)
        >>> steerer = DiscreteSteerer(generator)
        >>> # steer 3 times:
        >>> steered_desc = steerer.steer_descriptions(desc, steerer_power=3, normalize=True)
    """

    def __init__(self, generator: Tensor) -> None:
        super().__init__()
        self.generator = torch.nn.Parameter(generator)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, self.generator)

    def steer_descriptions(
        self,
        descriptions: Tensor,
        steerer_power: int = 1,
        normalize: bool = False,
    ) -> Tensor:
        for _ in range(steerer_power):
            descriptions = self.forward(descriptions)
        if normalize:
            descriptions = torch.nn.functional.normalize(descriptions, dim=-1)
        return descriptions

    @classmethod
    def create_dedode_default(
        cls,
        generator_type: str = "C4",
        steerer_order: int = 8,
    ) -> Module:
        r"""Creates a steerer for pretrained DeDoDe descriptors int the "C-setting"
            from the paper https://arxiv.org/abs/2312.02152, where descriptors were
            trained for fixed steerers.

        Args:
            generator_type: The type of steerer generator.
                One of 'C4', 'SO2', default is 'C4'.
                These can be used with the DeDoDe descriptors in Kornia
                with C4 or SO2 in the name respectively (so called C-setting steerers).
            steerer_order: The discretisation order for SO2-steerers (NOT used for C4-steerers).

        Returns:
            The pretrained model.
        """
        descriptor_dim = 256
        if generator_type == "C4":
            generator = torch.block_diag(
                *(
                    torch.tensor([[0.0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
                    for _ in range(descriptor_dim // 4)
                )
            )
            return cls(generator).eval()
        elif generator_type == "SO2":
            lie_generator = torch.block_diag(
                torch.zeros(
                    [descriptor_dim - 12 * (descriptor_dim // 14), descriptor_dim - 12 * (descriptor_dim // 14)],
                ),
                *(torch.tensor([[0.0, j], [-j, 0]]) for j in range(1, 7) for _ in range(descriptor_dim // 14)),
            )
            generator = torch.matrix_exp((2 * 3.14159 / steerer_order) * lie_generator)
            return cls(generator).eval()
        else:
            raise ValueError
