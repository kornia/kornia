from typing import Dict
import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.core import Tensor, tensor
from kornia.geometry.bbox import bbox_generator


class _BBoxBasedGenerator(RandomGeneratorBase):

    has_fit_batch_prob = True

    def fit_batch_prob(
        self, batch_shape: torch.Size, batch_prob: Tensor, params: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        to_apply = batch_prob.round()
        batch_size = batch_shape[0]
        size = tensor(
            (batch_shape[-2], batch_shape[-1]), device=params["dst"].device, dtype=params["dst"].dtype
        ).repeat(batch_size, 1)
        crop_src = bbox_generator(
            tensor([0] * batch_size, device=params["dst"].device, dtype=params["dst"].dtype),
            tensor([0] * batch_size, device=params["dst"].device, dtype=params["dst"].dtype),
            size[:, 1],
            size[:, 0],
        )
        crop_src = params["src"] * to_apply[:, None, None] + crop_src * (1 - to_apply[:, None, None])
        crop_dst = params["dst"] * to_apply[:, None, None] + crop_src * (1 - to_apply[:, None, None])
        output_size = params["output_size"] * to_apply[:, None] + params["input_size"] * (1 - to_apply[:, None])
        return dict(src=crop_src, dst=crop_dst, input_size=params["input_size"], output_size=output_size)
