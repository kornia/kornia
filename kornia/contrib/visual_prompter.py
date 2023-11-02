from __future__ import annotations

from typing import Any, Optional

import torch

from kornia.augmentation import LongestMaxSize
from kornia.augmentation.container.augment import AugmentationSequential
from kornia.contrib.models import Prompts, SegmentationResults
from kornia.contrib.models.sam import Sam, SamConfig
from kornia.core import Tensor, pad, tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.enhance import normalize
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints


class VisualPrompter:
    """This class allow the user to run multiple query with multiple prompts for a model.

    At the moment, we just support the SAM model. The model is loaded based on the given config.

    For default the images are transformed to have their long side with size of the `image_encoder.img_size`. This
    Prompter class ensure to transform the images and the prompts before prediction. Also, the image is passed
    automatically for the method `preprocess_image`, which is responsible for normalize the image and pad it to have
    the right size for the SAM model :math:`(\text{image_encoder.img_size}, \text{image_encoder.img_size})`. For
    default the image is normalized by the mean and standard deviation of the SAM dataset values.

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
        >>> boxes = Boxes(
        ...    torch.tensor(
        ...         [[[[0, 0], [0, 10], [10, 0], [10, 10]]]],
        ...         device=prompter.device,
        ...         dtype=torch.float32
        ...    ),
        ...    mode='xyxy'
        ... )
        >>> prediction = prompter.predict(boxes=boxes)
        >>> prediction.logits.shape
        torch.Size([1, 3, 256, 256])
    """

    def __init__(
        self,
        config: SamConfig = SamConfig(model_type="vit_h", pretrained=True),
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(config, SamConfig):
            self.model = Sam.from_config(config)
            transforms = (LongestMaxSize(self.model.image_encoder.img_size, p=1.0),)
            self.pixel_mean: Optional[Tensor] = tensor([123.675, 116.28, 103.53], device=device, dtype=dtype) / 255
            self.pixel_std: Optional[Tensor] = tensor([58.395, 57.12, 57.375], device=device, dtype=dtype) / 255
        else:
            raise NotImplementedError

        self.model = self.model.to(device=device, dtype=dtype)
        self.transforms = AugmentationSequential(*transforms, same_on_batch=True)

        self.device = device
        self.dtype = dtype
        self._original_image_size: None | tuple[int, int] = None
        self._input_image_size: None | tuple[int, int] = None
        self._input_encoder_size: None | tuple[int, int] = None
        self.reset_image()

    def preprocess_image(self, x: Tensor, mean: Optional[Tensor] = None, std: Optional[Tensor] = None) -> Tensor:
        """Normalize and pad a tensor.

        For normalize the tensor: will prioritize the `mean` and `std` passed as argument, if None will use the default
        Sam Dataset values.

        For pad the tensor: Will pad the tensor into the right and bottom to match with the size of
        `self.model.image_encoder.img_size`

        Args:
            x: The image to be preprocessed
            mean: Mean for each channel.
            std: Standard deviations for each channel.

        Returns:
            The image preprocessed (normalized if has mean and str available and padded to encoder size)
        """

        if isinstance(mean, Tensor) and isinstance(std, Tensor):
            x = normalize(x, mean, std)
        elif isinstance(self.pixel_mean, Tensor) and isinstance(self.pixel_std, Tensor):
            x = normalize(x, self.pixel_mean, self.pixel_std)

        encoder_im_size = self.model.image_encoder.img_size
        pad_h = encoder_im_size - x.shape[-2]
        pad_w = encoder_im_size - x.shape[-1]
        x = pad(x, (0, pad_w, 0, pad_h))

        return x

    @torch.no_grad()
    def set_image(self, image: Tensor, mean: Optional[Tensor] = None, std: Optional[Tensor] = None) -> None:
        """Set the embeddings from the given image with `image_decoder` of the model.

        Prepare the given image with the selected transforms and the preprocess method.

        Args:
            image: RGB image. Normally images with range of [0-1], the model preprocess normalize the
                   pixel values with the mean and std defined in its initialization. Expected to be into a float32
                   dtype. Shape :math:`(3, H, W)`.
        """
        KORNIA_CHECK_SHAPE(image, ["3", "H", "W"])

        self.reset_image()

        self._original_image_size = (image.shape[-2], image.shape[-1])

        image = self.transforms(image, data_keys=["input"])
        self._tfs_params = self.transforms._params
        self._input_image_size = (image.shape[-2], image.shape[-1])

        image = self.preprocess_image(image, mean, std)

        self._input_encoder_size = (image.shape[-2], image.shape[-1])

        self.image_embeddings = self.model.image_encoder(image)
        self.is_image_set = True

    def _valid_keypoints(self, keypoints: Keypoints | Tensor, labels: Tensor) -> Keypoints:
        """Validate the keypoints shape and ensure to be a Keypoints."""
        KORNIA_CHECK_SHAPE(keypoints.data, ["K", "N", "2"])
        KORNIA_CHECK_SHAPE(labels.data, ["K", "N"])
        KORNIA_CHECK(keypoints.shape[0] == labels.shape[0], "The keypoints and labels should have the same batch size")

        if isinstance(keypoints, Tensor):
            keypoints = Keypoints.from_tensor(keypoints)

        return keypoints

    def _valid_boxes(self, boxes: Boxes | Tensor) -> Boxes:
        """Validate the boxes shape and ensure to be a Boxes into xyxy mode."""
        if isinstance(boxes, Tensor):
            KORNIA_CHECK_SHAPE(boxes.data, ["K", "4"])
            boxes = Boxes(boxes, mode="xyxy")

        if boxes.mode == "xyxy":
            boxes_xyxy = boxes
        else:
            boxes_xyxy = Boxes(boxes.to_tensor(mode="xyxy"), mode="xyxy")

        return boxes_xyxy

    def _valid_masks(self, masks: Tensor) -> Tensor:
        """Validate the input masks shape."""
        KORNIA_CHECK_SHAPE(masks, ["K", "1", "256", "256"])
        return masks

    def _transform_prompts(
        self, *prompts: Tensor | Boxes | Keypoints, data_keys: list[str] = []
    ) -> dict[str, Tensor | Boxes | Keypoints]:
        transformed_prompts = self.transforms(*prompts, data_keys=data_keys, params=self._tfs_params)

        # prevent unpacking tensor when creating the output dict (issue #2627)
        if not isinstance(transformed_prompts, (list, tuple)):
            transformed_prompts = [transformed_prompts]
        return {key: transformed_prompts[idx] for idx, key in enumerate(data_keys)}

    def preprocess_prompts(
        self,
        keypoints: Optional[Keypoints | Tensor] = None,
        keypoints_labels: Optional[Tensor] = None,
        boxes: Optional[Boxes | Tensor] = None,
        masks: Optional[Tensor] = None,
    ) -> Prompts:
        """Validate and preprocess the given prompts to be aligned with the input image."""
        data_keys = []
        to_transform: list[Keypoints | Boxes | Tensor] = []

        if isinstance(keypoints, (Keypoints, Tensor)) and isinstance(keypoints_labels, Tensor):
            keypoints = self._valid_keypoints(keypoints, keypoints_labels)
            data_keys.append("keypoints")
            to_transform.append(keypoints)

        if isinstance(boxes, (Boxes, Tensor)):
            self._valid_boxes(boxes)
            data_keys.append("bbox_xyxy")
            to_transform.append(boxes)

        if isinstance(masks, Tensor):
            self._valid_masks(masks)

        data = self._transform_prompts(*to_transform, data_keys=data_keys)

        if "keypoints" in data and isinstance(data["keypoints"], Keypoints):
            kpts_tensor = data["keypoints"].to_tensor()
            if KORNIA_CHECK_IS_TENSOR(kpts_tensor) and KORNIA_CHECK_IS_TENSOR(keypoints_labels):
                points = (kpts_tensor, keypoints_labels)
        else:
            points = None

        if "bbox_xyxy" in data and isinstance(data["bbox_xyxy"], Boxes):
            _bbox = data["bbox_xyxy"].to_tensor(mode="xyxy")
            if KORNIA_CHECK_IS_TENSOR(_bbox):
                bbox = _bbox
        else:
            bbox = None

        return Prompts(points=points, boxes=bbox, masks=masks)

    @torch.no_grad()
    def predict(
        self,
        keypoints: Optional[Keypoints | Tensor] = None,
        keypoints_labels: Optional[Tensor] = None,
        boxes: Optional[Boxes | Tensor] = None,
        masks: Optional[Tensor] = None,
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
        options: dict[Any, Any] = {},
        disable: bool = False,
    ) -> None:
        """Applies `torch.compile(...)`/dynamo API into the VisualPrompter API.

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
            >>> # prompter.compile() # You should have torch >= 2.0.0 installed
            >>> # Use the prompter methods ...
        """

        # self.set_image = torch.compile(  # type: ignore[method-assign]
        #     self.set_image,
        #     fullgraph=fullgraph,
        #     dynamic=dynamic,
        #     backend=backend,
        #     mode=mode,
        #     options=options,
        #     disable=disable,
        # )
        # FIXME: compile set image will try to compile AugmentationSequential which fails
        self.model.image_encoder = torch.compile(  # type: ignore
            self.model.image_encoder,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )

        # self.preprocess_image = torch.compile(  # type: ignore[method-assign]
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
        # self.predict = torch.compile(  # type: ignore[method-assign]
        #     self.predict,
        #     fullgraph=fullgraph,
        #     dynamic=dynamic,
        #     backend=backend,
        #     mode=mode,
        #     options=options,
        #     disable=disable,
        # )
        self.model.mask_decoder = torch.compile(  # type: ignore
            self.model.mask_decoder,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        self.model.prompt_encoder = torch.compile(  # type: ignore
            self.model.prompt_encoder,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
