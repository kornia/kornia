from typing import Dict

import torch

from kornia.core import Module, Tensor
from kornia.utils.helpers import map_location_to_cpu

dedode_steerer_urls: Dict[str, Dict[str, str]] = {
    "B-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_steerer_setting_C.pth",
    "B-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_steerer_setting_C.pth",
    "G-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_C4_Perm_steerer_setting_C.pth",
    "G-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_SO2_Spread_steerer_setting_C.pth",
}


class DiscreteSteerer(Module):
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
    def from_pretrained(
        cls,
        generator_weights: str = "G-C4",
        steerer_order: int = None,
    ) -> Module:
        r"""Loads a pretrained DeDoDe steerer from the paper https://arxiv.org/abs/2312.02152.

        Args:
            generator_weights: The weights to load for the steerer generator.
                One of 'B-C4', 'B-SO2', 'G-C4', 'G-SO2', default is 'G-C4'.
            steerer_order: The discretisation order for SO2-steerers (NOT used for C4-steerers).

        Returns:
            The pretrained model.
        """
        if "C4" in generator_weights:
            generator = torch.hub.load_state_dict_from_url(
                dedode_steerer_urls[generator_weights],
                map_location=map_location_to_cpu,
            )
            return DiscreteSteerer(generator).eval()
        elif "SO2" in generator_weights:
            lie_generator = torch.hub.load_state_dict_from_url(
                dedode_steerer_urls[generator_weights],
                map_location=map_location_to_cpu,
            )
            generator = torch.matrix_exp((2 * 3.14159 / steerer_order) * lie_generator)
            return DiscreteSteerer(generator).eval()
        else:
            raise ValueError
