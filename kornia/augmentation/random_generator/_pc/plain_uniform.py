from typing import Dict, Tuple

from kornia.augmentation.random_generator._2d.plain_uniform import PlainUniformGenerator
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["PlainUniformGeneratorPC"]


class PlainUniformGeneratorPC(PlainUniformGenerator):
    r"""Generate random parameters that distributed uniformly.

    Args:
        *samplers: a list of tuple in a pattern of ``(factor, name, center, range)``, in which
            the factor can be a two-numbered tuple, or a ``(2,)`` shaped torch tensor. The name
            will be the corresponding key of the returning dict. The center and range must be
            both provided worked as a validator to the given factor.

    Returns:
        A dict of parameters to be passed for transformation according the number of samplers
        and the pointed returning name of each tuple.
            - ``name``: element-wise probabilities with a shape of (B, N).
    """

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size, N = batch_shape[0], batch_shape[1]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([t for t, _, _, _ in self.samplers])

        return {
            name: _adapted_rsampling((batch_size, N), dist, same_on_batch).to(device=_device, dtype=_dtype)
            for name, dist in self.sampler_dict.items()
        }
