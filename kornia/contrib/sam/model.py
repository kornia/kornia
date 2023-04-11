from __future__ import annotations

from dataclasses import dataclass

from kornia.core import Tensor


@dataclass
class SamPrediction:
    """To map the results of the `SamPredictor`

    Args:
        masks: Shape must be :math:`(B, K, H, W)` or :math:`(K, H, W)` where K is the number masks predicted.
        scores: Intersection Over Union for each prediction. Shape :math:`(B, K)` or :math:`(K)`.
        logits: These low resolution logits can be passed to a subsequent iteration as mask input. Shape of
                :math:`(B, K, H, W)` or :math:`(K, H, W)`, normally H=W=256.
    """

    masks: Tensor
    scores: Tensor
    logits: Tensor

    def drop(self, idx: int | slice | Tensor) -> SamPrediction:
        """Drop the passed index for all data.

        Performs `self.prop = self.prop[idx]` for each property
        """
        self.masks = self.masks[idx]
        self.scores = self.scores[idx]
        self.logits = self.logits[idx]

        return self
