from __future__ import annotations

import torch

from kornia.augmentation import AugmentationSequential, LongestMaxSize
from kornia.core import Device, Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from .architecture import Sam

# from kornia.core.check import KORNIA_CHECK
# from kornia.io import ImageLoadType, load_image


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
        # self.reset_image()

    # def reset_image(self) -> None:
    #     """Resets the currently set image."""
    #     self.is_image_set = False
    #     self.features = None
    #     self.orig_h = None
    #     self.orig_w = None
    #     self.input_h = None
    #     self.input_w = None

    def device(self) -> Device:
        return self.model.device

    #     @torch.no_grad()
    #     def set_image(self, image: Tensor | str, image_format: ImageLoadType = ImageLoadType.RGB8) -> None:
    #         """Calculates the image embeddings for the provided image, allowing masks to be predicted with the
    #         'predict' method.
    #
    #         Args:
    #             image: The tensor of the image into RGB format, or the path for the image
    #             image_format: The image format
    #         """
    #         KORNIA_CHECK(image_format == ImageLoadType.RGB8, 'We just supports 8bits images on RGB format.')
    #         if isinstance(image, str):
    #             img = load_image(image, image_format, self.device)[None, ...]
    #         else:
    #             img = image[None, ...]
    #
    #         KORNIA_CHECK_SHAPE(img, ['B', '3', 'H', 'W'])
    #         self.reset_image()
    #         self.original_size = tuple(img.shape[-2:])
    #
    #         # Transform the image to the form expected by the model
    #         input_image = self.transform.apply_image(img)
    #
    #         self.input_size = tuple(input_image.shape[-2:])
    #         KORNIA_CHECK(max(self.input_size) == self._longside_size)
    #
    #         self.features = self.model.image_encoder(input_image)
    #         self.is_image_set = True

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
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict masks for the given input prompts, using the currently set image.

        Input prompts are batched tensors and are expected to already be resized

        # TODO: docstring
        """
        KORNIA_CHECK_SHAPE(image, ['B', '3', 'H', 'W'])

        dk = ['input']
        _args = (image.type(torch.float32),)

        if isinstance(point_coords, Keypoints) and isinstance(point_labels, Tensor):
            KORNIA_CHECK_SHAPE(point_coords, ['B', 'N', '2'])
            KORNIA_CHECK_SHAPE(point_labels, ['B', 'N'])
            dk += ['keypoints']
            _args += (point_coords,)
            # points = (point_coords, point_labels)

        if isinstance(boxes, Boxes):
            dk += ['boxes']
            _args += (boxes,)

        if isinstance(mask_input, Tensor):
            KORNIA_CHECK_SHAPE(mask_input, ['B', '3', '256', '256'])

        # TODO: figure out a better way to transform when we can have missing datakeys
        tfs = AugmentationSequential(*self.transforms, data_keys=dk, same_on_batch=True)
        _tf_data = tfs(*_args)
        data = {k: _tf_data[idx] for idx, k in enumerate(dk)}
        points = (data['keypoints'].to_tensor(), point_labels) if 'keypoints' in data else None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points, boxes=data.get('boxes', None), masks=mask_input
        )

        # Predict masks
        input_image = self.model.preprocess(data['input'])[None, ...]  # FIXME: Why need to add batch dim here?
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.model.image_encoder(input_image),
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, tuple(input_image.shape[-2:]), tuple(image.shape[-2:]))

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks
