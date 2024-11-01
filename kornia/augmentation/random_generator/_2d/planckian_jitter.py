
import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check, _range_bound

__all__ = ["PlanckianJitterGenerator"]


class PlanckianJitterGenerator(RandomGeneratorBase):
    r"""Generate random planckian jitter parameters for a batch of images."""

    def __init__(self, domain: list[float]) -> None:
        super().__init__()
        self.domain = domain

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        idx_range = _range_bound(self.domain, "idx_range", device=device, dtype=dtype)

        _joint_range_check(idx_range, "idx_range", (0, self.domain[1]))
        self.pl_idx_dist = UniformDistribution(idx_range[0], idx_range[1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        pl_idx = _adapted_rsampling((batch_size,), self.pl_idx_dist, same_on_batch)

        return {"idx": pl_idx.long()}
