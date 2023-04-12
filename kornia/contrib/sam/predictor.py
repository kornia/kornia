from __future__ import annotations

from typing import Any

import torch

from kornia.augmentation import AugmentationSequential, LongestMaxSize
from kornia.contrib.sam.architecture import Sam
from kornia.contrib.sam.base import SegmentationResults
from kornia.core import Tensor, pad
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints


class SamPredictor:
    """Uses SAM to calculate the image embedding for an image, and then allow repeated, efficient mask prediction
    given prompts.

    Args:
        sam_model: The model to use for mask prediction.
    """

    def __init__(self, sam_model: Sam) -> None:
        super().__init__()
        self.model = sam_model
        self._longside_size = self.model.image_encoder.img_size
        transforms = (LongestMaxSize(self._longside_size, p=1.0),)
        self.tfs = AugmentationSequential(*transforms, same_on_batch=True)

    def preprocess(self, x: Tensor) -> Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.model.pixel_mean) / self.model.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        x = pad(x, (0, padw, 0, padh))
        return x

    @torch.no_grad()
    def __call__(
        self,
        image: Tensor,
        point_coords: Keypoints | Tensor | None = None,
        point_labels: Tensor | None = None,
        boxes: Boxes | Tensor | None = None,
        mask_input: Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> SegmentationResults:
        """Predict masks for the given image based on the input prompts.

        Args:
            image: Batch of RGB image. Normally 8bits images (range of [0-255]), the model preprocess normalize the
                   pixel values with the mean and std defined in its initialization. Expected to be into a float dtype.
                   Shape :math:`(3, H, W)`.
            point_coords: Point prompts to the model. Each point is in (X,Y) in pixels. Shape :math:`(K, N, 2)`. Where
                          `N` is the number of points and `K` the number of prompts.
            point_labels: Labels for the point prompts. 1 indicates a foreground point and 0 indicates a background
                          point. Shape :math:`(K, N, 2)`. Where `N` is the number of points, and `K` the number of
                          prompts.
            boxes: A box prompt to the model. If a tensor, should be in a xyxy mode. Shape :math:`(K, 4)`
            mask_input:  A low resolution mask input to the model, typically coming from a previous prediction
                         iteration. Has shape :math:`(K, 1, H, W)`, where for SAM, H=W=256.
            multimask_output: If true, the model will return three masks. For ambiguous input prompts (such as a
                              single click), this will often produce better masks than a single prediction. If only
                              a single mask is needed, the model's predicted quality score can be used to select the
                              best mask. For non-ambiguous prompts, such as multiple input prompts,
                              multimask_output=False can give better results.

        Returns:
            A prediction with the masks, scores (IoU of each predicted mask), and the low resolution logits.
        """
        KORNIA_CHECK_SHAPE(image, ['3', 'H', 'W'])

        dk = ['input']
        _args: tuple[Any, ...] = (image.type(torch.float32),)

        if isinstance(point_coords, (Keypoints, Tensor)) and isinstance(point_labels, Tensor):
            KORNIA_CHECK_SHAPE(point_coords.data, ['K', 'N', '2'])
            KORNIA_CHECK_SHAPE(point_labels.data, ['K', 'N'])
            if isinstance(point_coords, Tensor):
                point_coords = Keypoints.from_tensor(point_coords)

            dk += ['keypoints']
            _args += (point_coords,)

        if isinstance(boxes, (Boxes, Tensor)):
            if isinstance(boxes, Tensor):
                boxes = Boxes(boxes, mode='xyxy')

            boxes_xyxy = boxes if boxes.mode == 'xyxy' else Boxes(boxes.to_tensor(mode='xyxy'), mode='xyxy')

            dk += ['bbox_xyxy']
            _args += (boxes_xyxy,)

        if isinstance(mask_input, Tensor):
            KORNIA_CHECK_SHAPE(mask_input, ['K', '1', '256', '256'])

        _tf_data = self.tfs(*_args, data_keys=dk)
        data = {k: _tf_data[idx] for idx, k in enumerate(dk)}

        points = (data['keypoints'].to_tensor(), point_labels) if 'keypoints' in data else None
        bbox = data['bbox_xyxy'].to_tensor(mode='xyxy') if 'bbox_xyxy' in data else None
        image_input = self.preprocess(data['input'])

        prediction = self.model(
            image_input, [{'points': points, 'boxes': bbox, 'mask_inputs': mask_input}], multimask_output
        )[-1]

        # Upscale the masks to the original image resolution
        input_size = (data['input'].shape[-2], data['input'].shape[-1])
        original_size = (image.shape[-2], image.shape[-1])
        image_size_encoder = (self.model.image_encoder.img_size, self.model.image_encoder.img_size)
        prediction.original_res_logits(input_size, original_size, image_size_encoder)

        prediction = prediction.squeeze(0)

        return prediction
