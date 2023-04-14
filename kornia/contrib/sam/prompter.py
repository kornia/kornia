from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from kornia.augmentation import AugmentationSequential
from kornia.contrib import Sam
from kornia.contrib.sam.base import SegmentationResults
from kornia.core import Module, Tensor, pad
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.enhance import normalize
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

# class PromptKey(Enum):
#     keypoints = 1
#     boxes = 2
#     masks = 3
#     text = 4


@dataclass
class Prompts:
    points: tuple[Tensor, Tensor] | None = None
    boxes: Tensor | None = None
    masks: Tensor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.keypoints, Tensor) and isinstance(self.boxes, Tensor):
            KORNIA_CHECK(self.keypoints.shape[0] == self.boxes.shape[0], 'The prompts should have the same batch size!')

    @property
    def keypoints(self) -> Tensor | None:
        return self.points[0] if isinstance(self.points, tuple) else None

    @property
    def keypoints_labels(self) -> Tensor | None:
        return self.points[1] if isinstance(self.points, tuple) else None


class ImagePrompter:
    """This class uses a given model to generate Segmentations results from a batch of prompts."""

    def __init__(self, model: Sam, transforms: Sequence[Module]):
        self.model = model
        self.transforms = AugmentationSequential(*transforms, same_on_batch=True)

    def preprocess(self, x: Tensor) -> Tensor:
        if hasattr(self.model, 'pixel_mean') and hasattr(self.model, 'pixel_std'):
            x = normalize(x, self.model.pixel_mean.view(-1), self.model.pixel_std.view(-1))

        if hasattr(self.model.image_encoder, 'img_size'):
            encoder_im_size = self.model.image_encoder.img_size
            pad_h = encoder_im_size - x.shape[-2]
            pad_w = encoder_im_size - x.shape[-1]
            x = pad(x, (0, pad_w, 0, pad_h))

        return x

    @torch.no_grad()
    def set_image(self, image: Tensor) -> Tensor:
        KORNIA_CHECK_SHAPE(image, ['3', 'H', 'W'])

        self._original_image_size = (image.shape[-2], image.shape[-1])

        image = self.transforms(image, data_keys=['input'])
        self._tfs_params = self.transforms._params
        self._input_image_size = (image.shape[-2], image.shape[-1])

        image = self.preprocess(image)
        self._input_encoder_size = (image.shape[-2], image.shape[-1])

        self.image_embeddings = self.model.image_encoder(image)
        self.is_image_set = True

    def _check_keypoints(self, keypoints: Keypoints | Tensor, labels: Tensor) -> Keypoints:
        KORNIA_CHECK_SHAPE(keypoints.data, ['K', 'N', '2'])
        KORNIA_CHECK_SHAPE(labels.data, ['K', 'N'])
        KORNIA_CHECK(keypoints.shape[0] == labels.shape[0], 'The keypoints and labels should have the same batch size')

        if isinstance(keypoints, Tensor):
            keypoints = Keypoints.from_tensor(keypoints)

        return keypoints

    def _check_boxes(self, boxes: Boxes | Tensor) -> Boxes:
        if isinstance(boxes, Tensor):
            KORNIA_CHECK_SHAPE(boxes.data, ['K', '4'])
            boxes = Boxes(boxes, mode='xyxy')

        if boxes.mode == 'xyxy':
            boxes_xyxy = boxes
        else:
            boxes_xyxy = Boxes(boxes.to_tensor(mode='xyxy'), mode='xyxy')

        return boxes_xyxy

    def _check_masks(self, masks: Tensor) -> Tensor:
        KORNIA_CHECK_SHAPE(masks, ['K', '1', '256', '256'])
        return masks

    def _transform_prompts(
        self, *prompts: Tensor | Boxes | Keypoints, data_keys: list[str] = []
    ) -> dict[str, Tensor | Boxes | Keypoints]:
        transformed_prompts = self.transforms(*prompts, data_keys=data_keys, params=self._tfs_params)
        return {key: transformed_prompts[idx] for idx, key in enumerate(data_keys)}

    def prepare_prompts(
        self,
        keypoints: Keypoints | Tensor | None = None,
        keypoints_labels: Tensor | None = None,
        boxes: Boxes | Tensor | None = None,
        masks: Tensor | None = None,
    ) -> Prompts:
        data_keys = []
        to_transform: list[Keypoints | Boxes | Tensor, ...] = []

        if isinstance(keypoints, (Keypoints, Tensor)) and isinstance(keypoints_labels, Tensor):
            keypoints = self._check_keypoints(keypoints, keypoints_labels)
            data_keys.append('keypoints')
            to_transform.append(keypoints)

        if isinstance(boxes, (Boxes, Tensor)):
            self._check_boxes(boxes)
            data_keys.append('bbox_xyxy')
            to_transform.append(boxes)

        if isinstance(masks, Tensor):
            self._check_masks(masks)

        data = self._transform_prompts(*to_transform, data_keys=data_keys)

        points = (data['keypoints'].to_tensor()[None, ...], keypoints_labels) if 'keypoints' in data else None
        bbox = data['bbox_xyxy'].to_tensor(mode='xyxy') if 'bbox_xyxy' in data else None

        return Prompts(points=points, boxes=bbox, masks=masks)

    @torch.no_grad()
    def predict(
        self,
        keypoints: Keypoints | Tensor | None = None,
        keypoints_labels: Tensor | None = None,
        boxes: Boxes | Tensor | None = None,
        masks: Tensor | None = None,
        multimask_output: bool = True,
        output_original_size: bool = True,
    ) -> SegmentationResults:
        KORNIA_CHECK(self.is_image_set, 'An image must be set with `self.set_image(...)` before `predict` be called!')

        prompts = self.prepare_prompts(keypoints, keypoints_labels, boxes, masks)

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=prompts.points, boxes=prompts.boxes, masks=prompts.masks
        )
        del prompts

        # Predict masks
        logits, scores = self.model.mask_decoder(
            image_embeddings=self.image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        results = SegmentationResults(logits, scores)
        if output_original_size:
            results.original_res_logits(self._input_image_size, self._original_image_size, self._input_encoder_size)

        results = results.squeeze(0)

        return results
