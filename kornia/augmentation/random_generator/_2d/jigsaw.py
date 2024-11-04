from typing import Dict, Tuple

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.random_generator.utils import randperm
from kornia.augmentation.utils import _common_param_check

__all__ = ["JigsawGenerator"]


class JigsawGenerator(RandomGeneratorBase):
    r"""Generate Jigsaw permutation indices for a batch of inputs.

    Args:
        grid: the Jigsaw puzzle grid. e.g. (2, 2) means
            each output will mix image patches in a 2x2 grid.

    Returns:
        A dict of parameters to be passed for transformation.
            - permutation (Tensor): Jigsaw permutation arrangement.

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, grid: Tuple[int, int] = (4, 4), ensure_perm: bool = True) -> None:
        super().__init__()
        self.grid = grid
        self.ensure_perm = ensure_perm

    def __repr__(self) -> str:
        repr = f"grid={self.grid}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._device = device
        self._dtype = dtype

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)

        perm_times = self.grid[0] * self.grid[1]
        # Generate mosiac order in one shot
        if batch_size == 0:
            rand_ids = torch.zeros([0, perm_times], device=self._device)
        elif same_on_batch:
            rand_ids = randperm(perm_times, ensure_perm=self.ensure_perm, device=self._device)
            rand_ids = torch.stack([rand_ids] * batch_size)
        else:
            rand_ids = torch.stack(
                [randperm(perm_times, ensure_perm=self.ensure_perm, device=self._device) for _ in range(batch_size)]
            )
        return {"permutation": rand_ids}
