from __future__ import annotations

import torch

from kornia.augmentation import AugmentationSequential, LongestMaxSize
from kornia.core import Device, Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from .architecture import Sam
from .model import SamPrediction


class SamPredictor:
    def __init__(self, sam_model: Sam) -> None:
        """Uses SAM to calculate the image embedding for an image, and then allow repeated, efficient mask
        prediction given prompts.

        Args:
            sam_model: The model to use for mask prediction.
        """

        super().__init__()
        self.model = sam_model
        self._longside_size = self.model.image_encoder.img_size
        self.transforms = (LongestMaxSize(self._longside_size),)

    def device(self) -> Device:
        return self.model.device

    @torch.no_grad()
    def __call__(
        self,
        image: Tensor,
        point_coords: Keypoints | None = None,
        point_labels: Tensor | None = None,
        boxes: Boxes | None = None,
        mask_input: Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> SamPrediction:
        """Predict masks for the given image based on the input prompts.

        Args:
            image: Batch of RGB image. Normally 8bits images (range of [0-255]), the model preprocess normalize the
            pixel values with the mean and std defined in its initialization. Expected to be into a float dtype. Shape
            :math:`(B, 3, H, W)` or :math:`(3, H, W)`.
            point_coords: Point prompts to the model. Each point is in (X,Y) in pixels. Shape :math:`(B, N, 2)` or
            :math:`(N, 2)`. Where `N` is the number of points.
            point_labels: Labels for the point prompts. 1 indicates a foreground point and 0 indicates a background
            point. Shape :math:`(B, N)` or :math:`(N)`. Where `N` is the number of points.
            boxes: A box prompt to the model.
            mask_input:  A low resolution mask input to the model, typically coming from a previous prediction
            iteration. Has shape :math:`(B, 1, H, W)` or :math:`(1, H, W)`, where for SAM, H=W=256.
            multimask_output: If true, the model will return three masks. For ambiguous input prompts (such as a
            single click), this will often produce better masks than a single prediction. If only a single mask is
            needed, the model's predicted quality score can be used to select the best mask. For non-ambiguous prompts,
            such as multiple input prompts, multimask_output=False can give better results.
            return_logits: If true, returns un-thresholded masks logits instead of a binary mask.

        Returns:
            A prediction with the masks, scores (IoU of each predicted mask), and the low resolution logits.
        """
        KORNIA_CHECK_SHAPE(
            image, ['*', '3', 'H', 'W']
        )  # FIXME: should be [('B', None), '3', 'H', 'W'], same for the others

        dk = ['input']
        _args = (image.type(torch.float32),)

        if isinstance(point_coords, (Keypoints, Tensor)) and isinstance(point_labels, Tensor):
            KORNIA_CHECK_SHAPE(point_coords, ['*', 'N', '2'])
            KORNIA_CHECK_SHAPE(point_labels, ['*', 'N'])
            dk += ['keypoints']
            _args += (point_coords,)

        if isinstance(boxes, Boxes):
            if boxes.mode == 'xyxy':
                dk += ['bbox_xyxy']
                _args += (boxes,)
            else:
                raise NotImplementedError('We expects the boxes to be into xyxy mode.')

        if isinstance(mask_input, Tensor):
            KORNIA_CHECK_SHAPE(mask_input, ['*', '1', '256', '256'])

        # TODO: figure out a better way to transform when we can have missing datakeys
        tfs = AugmentationSequential(*self.transforms, data_keys=dk, same_on_batch=True)
        # FIXME: the augmentation sequential isn't accepting the k.geometry.Keypoints
        _tf_data = tfs(*_args)
        data = {k: _tf_data[idx] for idx, k in enumerate(dk)}

        points = (data['keypoints'], point_labels) if 'keypoints' in data else None
        bb = data['bbox_xyxy'].to_tensor(mode='xyxy') if 'bbox_xyxy' in data else None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=bb, masks=mask_input)

        # Predict masks
        input_image = self.model.preprocess(data['input'])
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.model.image_encoder(input_image),
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, tuple(data['input'].shape[-2:]), tuple(image.shape[-2:]))

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return SamPrediction(masks, iou_predictions, low_res_masks)
