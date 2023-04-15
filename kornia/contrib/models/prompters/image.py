from typing import Any

from torch import no_grad

from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.container.augment import AugmentationSequential
from kornia.contrib.models import Prompts, SegmentationResults
from kornia.contrib.models.base import ModelType
from kornia.contrib.models.prompters.base import PrompterModelBase
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints


class ImagePrompter:
    """This class uses a given model to generate Segmentations results from a batch of prompts.

    Args:
        model: The desired model to be used to generate results from prompts
        *transforms: The augmentation transforms desired to be performed on the input before prediction
    """

    def __init__(self, model: PrompterModelBase[ModelType], *transforms: _AugmentationBase):
        self.model = model
        self.transforms = AugmentationSequential(*transforms, same_on_batch=True)
        self.pixel_mean: Tensor | None = None
        self.pixel_std: Tensor | None = None

        self._original_image_size: None | tuple[int, int] = None
        self._input_image_size: None | tuple[int, int] = None
        self._input_encoder_size: None | tuple[int, int] = None
        self.reset_image()

    @no_grad()
    def set_image(self, image: Tensor, *args: Any, **kwargs: Any) -> None:
        """Set the embeddings from the given image with `image_decoder` of the model.

        Prepare the given image with the selected transforms and the preprocess method.

        Args:
            image: RGB image. Normally 8bits images (range of [0-255]), the model preprocess normalize the
                   pixel values with the mean and std defined in its initialization. Expected to be into a float dtype.
                   Shape :math:`(3, H, W)`.
        """
        KORNIA_CHECK_SHAPE(image, ['3', 'H', 'W'])

        self.reset_image()

        self._original_image_size = (image.shape[-2], image.shape[-1])

        image = self.transforms(image, data_keys=['input'])
        self._tfs_params = self.transforms._params
        self._input_image_size = (image.shape[-2], image.shape[-1])

        if hasattr(self, 'preprocess_image'):
            image = self.preprocess_image(image, *args, **kwargs)

        self._input_encoder_size = (image.shape[-2], image.shape[-1])

        self.image_embeddings = self.model.image_encoder(image)
        self.is_image_set = True

    def _valid_keypoints(self, keypoints: Keypoints | Tensor, labels: Tensor) -> Keypoints:
        """Validate the keypoints shape and ensure to be a Keypoints."""
        KORNIA_CHECK_SHAPE(keypoints.data, ['K', 'N', '2'])
        KORNIA_CHECK_SHAPE(labels.data, ['K', 'N'])
        KORNIA_CHECK(keypoints.shape[0] == labels.shape[0], 'The keypoints and labels should have the same batch size')

        if isinstance(keypoints, Tensor):
            keypoints = Keypoints.from_tensor(keypoints)

        return keypoints

    def _valid_boxes(self, boxes: Boxes | Tensor) -> Boxes:
        """Validate the boxes shape and ensure to be a Boxes into xyxy mode."""
        if isinstance(boxes, Tensor):
            KORNIA_CHECK_SHAPE(boxes.data, ['K', '4'])
            boxes = Boxes(boxes, mode='xyxy')

        if boxes.mode == 'xyxy':
            boxes_xyxy = boxes
        else:
            boxes_xyxy = Boxes(boxes.to_tensor(mode='xyxy'), mode='xyxy')

        return boxes_xyxy

    def _valid_masks(self, masks: Tensor) -> Tensor:
        """Validate the input masks shape."""
        KORNIA_CHECK_SHAPE(masks, ['K', '1', '256', '256'])
        return masks

    def _transform_prompts(
        self, *prompts: Tensor | Boxes | Keypoints, data_keys: list[str] = []
    ) -> dict[str, Tensor | Boxes | Keypoints]:
        transformed_prompts = self.transforms(*prompts, data_keys=data_keys, params=self._tfs_params)
        return {key: transformed_prompts[idx] for idx, key in enumerate(data_keys)}

    def preprocess_prompts(
        self,
        keypoints: Keypoints | Tensor | None = None,
        keypoints_labels: Tensor | None = None,
        boxes: Boxes | Tensor | None = None,
        masks: Tensor | None = None,
    ) -> Prompts:
        """Validate and preprocess the given prompts to be aligned with the input image."""
        data_keys = []
        to_transform: list[Keypoints | Boxes | Tensor] = []

        if isinstance(keypoints, (Keypoints, Tensor)) and isinstance(keypoints_labels, Tensor):
            keypoints = self._valid_keypoints(keypoints, keypoints_labels)
            data_keys.append('keypoints')
            to_transform.append(keypoints)

        if isinstance(boxes, (Boxes, Tensor)):
            self._valid_boxes(boxes)
            data_keys.append('bbox_xyxy')
            to_transform.append(boxes)

        if isinstance(masks, Tensor):
            self._valid_masks(masks)

        data = self._transform_prompts(*to_transform, data_keys=data_keys)

        if 'keypoints' in data and isinstance(data['keypoints'], Keypoints):
            kpts_tensor = data['keypoints'].to_tensor()
            if KORNIA_CHECK_IS_TENSOR(kpts_tensor) and KORNIA_CHECK_IS_TENSOR(keypoints_labels):
                points = (kpts_tensor[None, ...], keypoints_labels)
        else:
            points = None

        if 'bbox_xyxy' in data and isinstance(data['bbox_xyxy'], Boxes):
            _bbox = data['bbox_xyxy'].to_tensor(mode='xyxy')
            if KORNIA_CHECK_IS_TENSOR(_bbox):
                bbox = _bbox
        else:
            bbox = None

        return Prompts(points=points, boxes=bbox, masks=masks)

    @no_grad()
    def predict(
        self,
        keypoints: Keypoints | Tensor | None = None,
        keypoints_labels: Tensor | None = None,
        boxes: Boxes | Tensor | None = None,
        masks: Tensor | None = None,
        multimask_output: bool = True,
        output_original_size: bool = True,
    ) -> SegmentationResults:
        """Predict masks for the given image based on the input prompts.

        Args:
            keypoints: Point prompts to the model. Each point is in (X,Y) in pixels. Shape :math:`(K, N, 2)`. Where
                       `N` is the number of points and `K` the number of prompts.
            keypoint_labels: Labels for the point prompts. 1 indicates a foreground point and 0 indicates a background
                             point. Shape :math:`(K, N)`. Where `N` is the number of points, and `K` the number of
                             prompts.
            boxes: A box prompt to the model. If a tensor, should be in a xyxy mode. Shape :math:`(K, 4)`
            masks: A low resolution mask input to the model, typically coming from a previous prediction
                   iteration. Has shape :math:`(K, 1, H, W)`, where for SAM, H=W=256.
            multimask_output: If true, the model will return three masks. For ambiguous input prompts (such as a
                              single click), this will often produce better masks than a single prediction. If only
                              a single mask is needed, the model's predicted quality score can be used to select the
                              best mask. For non-ambiguous prompts, such as multiple input prompts,
                              multimask_output=False can give better results.
            output_original_size: If true, the logits of `SegmentationResults` will be post-process to match the
                                  original input image size.
        Returns:
            A prediction with the logits and scores (IoU of each predicted mask)
        """
        KORNIA_CHECK(self.is_image_set, 'An image must be set with `self.set_image(...)` before `predict` be called!')

        prompts = self.preprocess_prompts(keypoints, keypoints_labels, boxes, masks)

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
        if (
            output_original_size
            and isinstance(self._input_image_size, tuple)
            and isinstance(self._original_image_size, tuple)
        ):
            results.original_res_logits(self._input_image_size, self._original_image_size, self._input_encoder_size)

        results = results.squeeze(0)

        return results

    def reset_image(self) -> None:
        self._tfs_params = None
        self._original_image_size = None
        self._input_image_size = None
        self._input_encoder_size = None

        if hasattr(self, 'image_embeddings'):
            del self.image_embeddings

        self.image_embeddings = None
        self.is_image_set = False
