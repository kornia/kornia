from __future__ import annotations

from dataclasses import dataclass

from kornia.core import Tensor
from kornia.geometry.transform import resize


@dataclass
class SegmentationResults:
    """Encapsulate the results obtained by a Segmentation model.

    Args:
        logits: Results logits with shape :math:`(B, C, H, W)`, where :math:`C` refers to the number of predicted masks
        scores: The scores from the logits. Shape :math:`(B,)`
        mask_threshold: The threshold value to generate the `binary_masks` from the `logits`
    """

    logits: Tensor
    scores: Tensor
    mask_threshold: float = 0.0

    @property
    def binary_masks(self) -> Tensor:
        """Binary mask generated from logits considering the mask_threshold.

        Shape will be the same of logits :math:`(B, C, H, W)` where :math:`C` is the number masks predicted.

        .. note: If you run `original_res_logits`, this will generate the masks based on the original resolution logits.
           Otherwise, this will use the low resolution logits (self.logits).
        """
        if self._original_res_logits is not None:
            x = self._original_res_logits
        else:
            x = self.logits

        return x > self.mask_threshold

    def original_res_logits(
        self, input_size: tuple[int, int], original_size: tuple[int, int], image_size_encoder: tuple[int, int] | None
    ) -> Tensor:
        """Remove padding and upscale the logits to the original image size.

        Resize to image encoder input -> remove padding (bottom and right) -> Resize to original size

        .. note: This method set a internal `original_res_logits` which will be used if available for the binary masks.

        Args:
            input_size: The size of the image input to the model, in (H, W) format. Used to remove padding.
            original_size: The original size of the image before resizing for input to the model, in (H, W) format.
            image_size_encoder: The size of the input image for image encoder, in (H, W) format. Used to resize the
                                logits back to encoder resolution before remove the padding.

        Returns:
            Batched logits in :math:`(K, C, H, W)` format, where (H, W) is given by original_size.
        """
        x = self.logits

        if isinstance(image_size_encoder, tuple):
            x = resize(x, size=image_size_encoder, interpolation='bilinear', align_corners=False, antialias=False)
        x = x[..., : input_size[0], : input_size[1]]

        x = resize(x, size=original_size, interpolation='bilinear', align_corners=False, antialias=False)

        self._original_res_logits = x
        return self._original_res_logits

    def squeeze(self, dim: int = 0) -> SegmentationResults:
        """Realize a squeeze for the dim given for all properties."""
        self.logits = self.logits.squeeze(dim)
        self.scores = self.scores.squeeze(dim)
        if isinstance(self._original_res_logits, Tensor):
            self._original_res_logits = self._original_res_logits.squeeze(dim)

        return self
