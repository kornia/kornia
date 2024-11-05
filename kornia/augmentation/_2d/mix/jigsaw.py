from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.constants import DataKey
from kornia.core import Tensor

__all__ = ["RandomJigsaw"]


class RandomJigsaw(MixAugmentationBaseV2):
    r"""RandomJigsaw augmentation.

    .. image:: _static/img/RandomJigsaw.png

    Make Jigsaw puzzles for each image individually. To mix with different images in a
    batch, referring to :class:`kornia.augmentation.RandomMosic`.

    Args:
        grid: the Jigsaw puzzle grid. e.g. (2, 2) means
            each output will mix image patches in a 2x2 grid.
        ensure_perm: to ensure the nonidentical patch permutation generation against
            the original one.
        data_keys: the input type sequential for applying augmentations.
            Accepts "input", "image", "mask", "bbox", "bbox_xyxy", "bbox_xywh", "keypoints",
            "class", "label".
        p: probability of applying the transformation for the whole batch.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
            to the batch form ``False``.

    Examples:
        >>> jigsaw = RandomJigsaw((4, 4))
        >>> input = torch.randn(8, 3, 256, 256)
        >>> out = jigsaw(input)
        >>> out.shape
        torch.Size([8, 3, 256, 256])
    """

    def __init__(
        self,
        grid: Tuple[int, int] = (4, 4),
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
        p: float = 0.5,
        same_on_batch: bool = False,
        keepdim: bool = False,
        ensure_perm: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=1.0, same_on_batch=same_on_batch, keepdim=keepdim, data_keys=data_keys)
        self._param_generator = rg.JigsawGenerator(grid, ensure_perm)
        self.flags = {"grid": grid}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], maybe_flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        # different from the Base class routine. This function will not refer to any non-transformation images.
        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.
        input = input[to_apply].clone()

        b, c, h, w = input.shape
        perm = params["permutation"]
        # Note: with a 100x100 image and a grid size of 3x3, it could work if
        #       we make h = piece_size_h * self.flags["grid"][0] with one pixel loss, then resize to 100 x 100.
        #       Probably worth to check if we should tolerate such "errorness" or to raise it as an error.
        piece_size_h, piece_size_w = input.shape[-2] // self.flags["grid"][0], input.shape[-1] // self.flags["grid"][1]
        # Convert to C BxN H' W'
        input = (
            input.unfold(2, piece_size_h, piece_size_h)
            .unfold(3, piece_size_w, piece_size_w)
            .reshape(b, c, -1, piece_size_h, piece_size_w)
            .permute(1, 0, 2, 3, 4)
            .reshape(c, -1, piece_size_h, piece_size_w)
        )
        perm = (perm + torch.arange(0, b, device=perm.device)[:, None] * perm.shape[1]).view(-1)
        input = input[:, perm, :, :]
        input = (
            input.reshape(-1, b, self.flags["grid"][1], h, piece_size_w)
            .permute(0, 1, 2, 4, 3)
            .reshape(-1, b, w, h)
            .permute(0, 1, 3, 2)
            .permute(1, 0, 2, 3)
        )
        return input
