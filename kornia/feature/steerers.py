import torch
from kornia.core import Module, Tensor


class DiscreteSteerer(Module):
    def __init__(self, generator: Tensor):
        super().__init__()
        self.generator = generator

    def forward(self, x: Tensor):
        return torch.nn.functional.linear(x, self.generator)

    def steer_descriptions(
        self, descriptions: Tensor, steerer_power: int = 1, normalize: bool = False,
    ):
        for _ in range(steerer_power):
            descriptions = self.forward(descriptions)
        if normalize:
            descriptions = torch.nn.functional.normalize(descriptions, dim=-1)
        return descriptions
