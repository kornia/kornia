# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from kornia.augmentation import Resize
from kornia.augmentation.container.augment import AugmentationSequential
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.enhance import normalize as kornia_normalize
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.models.sam import Sam, SamConfig
from kornia.models.structures import Prompts, SegmentationResults


class VisualPrompter:
    r"""Allow the user to run multiple queries with multiple prompts on one image.

    At the moment, we just support the SAM model. The model is loaded based on the given config.

    The prompter holds one image at a time: :meth:`set_image` takes a :math:`(3, H, W)` or :math:`(1, 3, H, W)` image,
    and :meth:`predict` then answers prompts on it. To segment a batch of images, each with its own prompts, use
    :meth:`kornia.models.sam.Sam.forward`.

    The image is resized so that its long side equals ``image_encoder.img_size``, by
    ``kornia.augmentation.Resize(img_size, side="long", align_corners=False, antialias=True)``: a bilinear resize on
    the half-pixel grid, antialiased when it downscales. Keypoint and box prompts go through the same transform. The
    image is then passed to the method `preprocess_image`, which normalizes it and pads it to the size the SAM model
    takes, :math:`(\text{image_encoder.img_size}, \text{image_encoder.img_size})`. For default the image is normalized
    by the mean and standard deviation of the SAM dataset values.

    Two details depart on purpose from Meta's ``SamPredictor`` (``segment-anything@3518c86``):

    - The resized short side is truncated, where the reference rounds it (a 1380 x 2076 image becomes 680 x 1024
      here and 681 x 1024 there). Truncation is the rule of :class:`kornia.augmentation.Resize`, and the masks are
      resized back through the same recorded input size.
    - Prompts follow the half-pixel map :math:`x \mapsto (x + 0.5)\,s - 0.5`, kornia's pixel-centre convention and the
      map the image resize uses, where the reference's ``apply_coords`` uses :math:`x \mapsto x\,s`. With the scale
      :math:`s = 8`, the keypoint (40, 30) goes to (323.5, 243.5) here and to (320, 240) in the reference.

    The held model follows the mode rule of :ref:`pretrained-models-conventions`: a config that loads weights gives a
    model in evaluation mode, and a randomly initialised one stays in training mode, where MobileSAM's BatchNorm
    normalises with the statistics of the batch. Call ``prompter.model.eval()`` before inference on random weights.

    Args:
        config: A model config to generate the model. Now just the SAM model is supported.
        device: The desired device to use the model.
        dtype: The desired dtype to use the model.

    Example:
        >>> # prompter = VisualPrompter() # Will load the vit h for default
        >>> # You can load a custom SAM type for modifying the config
        >>> prompter = VisualPrompter(SamConfig('vit_b'))
        >>> image = torch.rand(3, 25, 30)
        >>> prompter.set_image(image)
        >>> # The K box prompts of the image, as a (1, K, 4) xyxy tensor; here K = 1.
        >>> boxes = Boxes.from_tensor(
        ...     torch.tensor([[[0.0, 0.0, 10.0, 10.0]]], device=prompter.device, dtype=torch.float32), mode='xyxy'
        ... )
        >>> prediction = prompter.predict(boxes=boxes)
        >>> prediction.logits.shape
        torch.Size([1, 3, 256, 256])

    """

    def __init__(
        self,
        config: Optional[SamConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = SamConfig(model_type="vit_h", pretrained=True)

        if isinstance(config, SamConfig):
            self.model = Sam.from_config(config)
            transforms = (Resize(self.model.image_encoder.img_size, side="long", align_corners=False, antialias=True),)
            self.pixel_mean: Optional[torch.Tensor] = (
                torch.tensor([123.675, 116.28, 103.53], device=device, dtype=dtype) / 255
            )
            self.pixel_std: Optional[torch.Tensor] = (
                torch.tensor([58.395, 57.12, 57.375], device=device, dtype=dtype) / 255
            )
        else:
            raise NotImplementedError

        self.model = self.model.to(device=device, dtype=dtype)
        self.transforms = AugmentationSequential(*transforms, same_on_batch=True)

        self.device = device
        self.dtype = dtype
        self._original_image_size: tuple[int, int] | None = None
        self._input_image_size: tuple[int, int] | None = None
        self._input_encoder_size: tuple[int, int] | None = None
        self.reset_image()

    def preprocess_image(
        self, x: torch.Tensor, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Normalize and F.pad a torch.Tensor.

        For F.normalize the tensor: will prioritize the `mean` and `std` passed as argument,
        if None will use the default
        Sam Dataset values.

        For F.pad the tensor: Will F.pad the torch.Tensor into the right and bottom to match with the size of
        `self.model.image_encoder.img_size`

        Args:
            x: The image to be preprocessed
            mean: Mean for each channel.
            std: Standard deviations for each channel.

        Returns:
            The image preprocessed (normalized if has mean and str available and padded to encoder size)

        """
        if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            x = kornia_normalize(x, mean, std)
        elif isinstance(self.pixel_mean, torch.Tensor) and isinstance(self.pixel_std, torch.Tensor):
            x = kornia_normalize(x, self.pixel_mean, self.pixel_std)

        encoder_im_size = self.model.image_encoder.img_size
        pad_h = encoder_im_size - x.shape[-2]
        pad_w = encoder_im_size - x.shape[-1]

        return F.pad(x, (0, pad_w, 0, pad_h))

    @torch.no_grad()
    def set_image(
        self, image: torch.Tensor, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None
    ) -> None:
        """Set the embeddings from the given image with `image_decoder` of the model.

        Prepare the given image with the selected transforms and the preprocess method.

        Args:
            image: One RGB image, shape :math:`(3, H, W)` or :math:`(1, 3, H, W)`, in the dtype of the model. Normally
                   images with range of [0-1], the model preprocess F.normalize the pixel values with the mean and std
                   defined in its initialization. A batch of more than one image raises; use
                   :meth:`kornia.models.sam.Sam.forward` for batches.
            mean: mean value of dataset for normalization.
            std: standard deviation of dataset for normalization.

        """
        if image.ndim == 3:
            KORNIA_CHECK_SHAPE(image, ["3", "H", "W"])
            image = image.unsqueeze(0)
        else:
            KORNIA_CHECK_SHAPE(image, ["B", "3", "H", "W"])
            KORNIA_CHECK(
                image.shape[0] == 1,
                f"VisualPrompter takes one image, (3, H, W) or (1, 3, H, W); got a batch of {image.shape[0]}. "
                "Use `Sam.forward` to segment a batch of images with per-image prompts.",
            )

        self.reset_image()

        self._original_image_size = (image.shape[-2], image.shape[-1])

        image = self.transforms(image, data_keys=["input"])
        self._tfs_params = self.transforms._params
        self._input_image_size = (image.shape[-2], image.shape[-1])

        image = self.preprocess_image(image, mean, std)

        self._input_encoder_size = (image.shape[-2], image.shape[-1])

        self.image_embeddings = self.model.image_encoder(image)
        self.is_image_set = True

    def _valid_keypoints(self, keypoints: Keypoints | torch.Tensor, labels: torch.Tensor) -> Keypoints:
        """Validate the keypoints shape and ensure to be a Keypoints."""
        KORNIA_CHECK_SHAPE(keypoints.data, ["K", "N", "2"])
        KORNIA_CHECK_SHAPE(labels.data, ["K", "N"])
        KORNIA_CHECK(keypoints.shape[0] == labels.shape[0], "The keypoints and labels should have the same batch size")

        if isinstance(keypoints, torch.Tensor):
            keypoints = Keypoints.from_tensor(keypoints)

        return keypoints

    def _valid_boxes(self, boxes: Boxes | torch.Tensor) -> Boxes:
        """Validate the boxes shape and ensure to be a Boxes into xyxy mode.

        The boxes are the ``K`` box prompts of the one image set: a ``(K, 4)`` or ``(1, K, 4)`` xyxy tensor, or a
        :class:`~kornia.geometry.boxes.Boxes` holding ``(K, 4, 2)`` or ``(1, K, 4, 2)`` vertices.
        """
        if isinstance(boxes, torch.Tensor):
            if boxes.ndim == 2:
                KORNIA_CHECK_SHAPE(boxes.data, ["K", "4"])
            else:
                KORNIA_CHECK_SHAPE(boxes.data, ["B", "K", "4"])

            boxes = Boxes.from_tensor(boxes, mode="xyxy")

        # A batched Boxes stores (B, K, 4, 2) vertices, and a (B, K, 4) tensor becomes one.
        KORNIA_CHECK(
            boxes.data.ndim == 3 or boxes.data.shape[0] == 1,
            f"VisualPrompter takes the boxes of its one image, (K, 4) or (1, K, 4); got boxes for "
            f"{boxes.data.shape[0]} images. Use `Sam.forward` to prompt a batch of images.",
        )

        return boxes if boxes.mode == "xyxy" else Boxes.from_tensor(boxes.to_tensor(mode="xyxy"), mode="xyxy")

    def _valid_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Validate the input masks shape."""
        KORNIA_CHECK_SHAPE(masks, ["K", "1", "256", "256"])
        return masks

    def _transform_prompts(
        self, *prompts: torch.Tensor | Boxes | Keypoints, data_keys: Optional[list[str]] = None
    ) -> dict[str, torch.Tensor | Boxes | Keypoints]:
        transformed_prompts = self.transforms(*prompts, data_keys=data_keys, params=self._tfs_params)
        if data_keys is None:
            data_keys = []

        # prevent unpacking torch.Tensor when creating the output dict (issue #2627)
        if not isinstance(transformed_prompts, (list, tuple)):
            transformed_prompts = [transformed_prompts]
        return {key: transformed_prompts[idx] for idx, key in enumerate(data_keys)}

    def preprocess_prompts(
        self,
        keypoints: Optional[Keypoints | torch.Tensor] = None,
        keypoints_labels: Optional[torch.Tensor] = None,
        boxes: Optional[Boxes | torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Prompts:
        """Validate and preprocess the given prompts to be aligned with the input image."""
        data_keys = []
        to_transform: list[Keypoints | Boxes | torch.Tensor] = []

        if isinstance(keypoints, (Keypoints, torch.Tensor)) and isinstance(keypoints_labels, torch.Tensor):
            keypoints = self._valid_keypoints(keypoints, keypoints_labels)
            data_keys.append("keypoints")
            to_transform.append(keypoints)
        elif keypoints is not None:
            KORNIA_CHECK(
                False,
                "keypoints_labels must be given with keypoints: a (K, N) tensor, 1 for a foreground point and 0 for "
                "a background point.",
            )

        if isinstance(boxes, (Boxes, torch.Tensor)):
            boxes = self._valid_boxes(boxes)
            data_keys.append("bbox_xyxy")
            to_transform.append(boxes)

        if isinstance(masks, torch.Tensor):
            self._valid_masks(masks)

        # No prompt at all is a valid query (SAM then predicts from the image embedding alone); the
        # augmentation container cannot be called with nothing to transform.
        data = self._transform_prompts(*to_transform, data_keys=data_keys) if to_transform else {}

        if "keypoints" in data and isinstance(data["keypoints"], Keypoints):
            kpts_tensor = data["keypoints"].to_tensor()
            if KORNIA_CHECK_IS_TENSOR(kpts_tensor) and KORNIA_CHECK_IS_TENSOR(keypoints_labels):
                points = (kpts_tensor, keypoints_labels)
        else:
            points = None

        if "bbox_xyxy" in data and isinstance(data["bbox_xyxy"], Boxes):
            # (K, 4) or (1, K, 4) -> the K box prompts, (K, 4).
            _bbox = data["bbox_xyxy"].to_tensor(mode="xyxy").reshape(-1, 4)
            if KORNIA_CHECK_IS_TENSOR(_bbox):
                bbox = _bbox
        else:
            bbox = None

        return Prompts(points=points, boxes=bbox, masks=masks)

    @torch.no_grad()
    def predict(
        self,
        keypoints: Optional[Keypoints | torch.Tensor] = None,
        keypoints_labels: Optional[torch.Tensor] = None,
        boxes: Optional[Boxes | torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        output_original_size: bool = True,
    ) -> SegmentationResults:
        """Predict masks for the given image based on the input prompts.

        Args:
            keypoints: Point prompts to the model. Each point is in (X,Y) in pixels. Shape :math:`(K, N, 2)`. Where
                       `N` is the number of points and `K` the number of prompts.
            keypoints_labels: Labels for the point prompts. 1 indicates a foreground point and 0 indicates a background
                             point. Shape :math:`(K, N)`. Where `N` is the number of points, and `K` the number of
                             prompts. Required with `keypoints`.
            boxes: Box prompts to the model, one per prompt: a torch.Tensor in xyxy mode, shape :math:`(K, 4)` or
                   :math:`(1, K, 4)`, or a :class:`~kornia.geometry.boxes.Boxes` of shape :math:`(K, 4, 2)` or
                   :math:`(1, K, 4, 2)`. `K` is the number of prompts.
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
        KORNIA_CHECK(self.is_image_set, "An image must be set with `self.set_image(...)` before `predict` be called!")

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

        # results = results.squeeze(0)

        return results

    def reset_image(self) -> None:
        """Clear cached image state and prompt-transform metadata.

        This method invalidates previously computed image embeddings and resets all
        size/transform bookkeeping so a new call to :meth:`set_image` starts from a
        clean state.

        In practice, this resets:
        - transformed-image parameters,
        - original/input/encoder spatial sizes,
        - cached image embeddings,
        - ``is_image_set`` status flag.
        """
        self._tfs_params = None
        self._original_image_size = None
        self._input_image_size = None
        self._input_encoder_size = None

        if hasattr(self, "image_embeddings"):
            del self.image_embeddings

        self.image_embeddings = None
        self.is_image_set = False

    def compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
        mode: Optional[str] = None,
        options: Optional[dict[Any, Any]] = None,
        disable: bool = False,
    ) -> None:
        """Apply `torch.compile(...)`/dynamo API into the VisualPrompter API.

        .. note:: For more information about the dynamo API check the official docs
                  https://pytorch.org/docs/stable/generated/torch.compile.html

        Args:
            fullgraph: Whether it is ok to break model into several subgraphs
            dynamic: Use dynamic shape tracing
            backend: backend to be used
            mode: Can be either “default”, “reduce-overhead” or “max-autotune”
            options: A dictionary of options to pass to the backend.
            disable: Turn torch.compile() into a no-op for testing

        Example:
            >>> # prompter = VisualPrompter()
            >>> # prompter.compile()
            >>> # Use the prompter methods ...

        """
        # self.set_image = torch.compile(
        #     self.set_image,
        #     fullgraph=fullgraph,
        #     dynamic=dynamic,
        #     backend=backend,
        #     mode=mode,
        #     options=options,
        #     disable=disable,
        # )
        # FIXME: compile set image will try to compile AugmentationSequential which fails
        self.model.image_encoder = torch.compile(
            self.model.image_encoder,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )

        # self.preprocess_image = torch.compile(
        #     self.preprocess_image,
        #     fullgraph=fullgraph,
        #     dynamic=dynamic,
        #     backend=backend,
        #     mode=mode,
        #     options=options,
        #     disable=disable,
        # )

        # FIXME: compile predict will try to compile Preproc prompts, which need to compileAugmentationSequential
        # which fails
        # self.predict = torch.compile(
        #     self.predict,
        #     fullgraph=fullgraph,
        #     dynamic=dynamic,
        #     backend=backend,
        #     mode=mode,
        #     options=options,
        #     disable=disable,
        # )
        self.model.mask_decoder = torch.compile(
            self.model.mask_decoder,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        self.model.prompt_encoder = torch.compile(
            self.model.prompt_encoder,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
