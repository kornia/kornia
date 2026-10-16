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
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.enhance import normalize as kornia_normalize
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.models.sam import Sam, SamConfig
from kornia.models.structures import Prompts, SegmentationResults


def _resize_like_sam(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize as SAM's reference does: bilinear on the half-pixel grid, with torch's antialiasing.

    This is ``ResizeLongestSide.apply_image_torch`` of ``segment-anything@3518c86``. It runs in at least float32, as
    torch has no half-precision antialiased kernel on the CPU, and the result is cast back to the dtype of ``image``.
    """
    x = image.to(torch.promote_types(image.dtype, torch.float32))
    try:
        out = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    except NotImplementedError:
        if x.device.type != "mps":
            raise
        # torch 2.5.1 has no MPS kernel for the antialiased bilinear resize (``_upsample_bilinear2d_aa``).
        out = F.interpolate(x.cpu(), size=size, mode="bilinear", align_corners=False, antialias=True).to(x.device)
    return out.to(image.dtype)


class VisualPrompter:
    r"""Allow the user to run multiple queries with multiple prompts on one image.

    At the moment, we just support the SAM model. The model is loaded based on the given config.

    The prompter holds one image at a time: :meth:`set_image` takes a :math:`(3, H, W)` or :math:`(1, 3, H, W)` image,
    and :meth:`predict` then answers prompts on it. To segment a batch of images, each with its own prompts, use
    :meth:`kornia.models.sam.Sam.forward`.

    The image is resized so that its long side equals ``image_encoder.img_size``. The output size and the transform
    that maps the prompts come from ``transforms``,
    ``kornia.augmentation.Resize(img_size, side="long", align_corners=False, antialias=True)``. The image itself is
    resized as SAM's reference resizes it (``ResizeLongestSide.apply_image_torch``), with
    ``torch.nn.functional.interpolate(image, size, mode="bilinear", align_corners=False, antialias=True)``, computed in
    float32 (float64 for a float64 image) and cast back. Keypoints and both corners of each box prompt go through the
    half-pixel map of ``transforms``, the map this resize samples on. The image is then passed to the method
    `preprocess_image`, which normalizes it and pads it to the size the SAM model takes,
    :math:`(\text{image_encoder.img_size}, \text{image_encoder.img_size})`. For default the image is normalized by the
    mean and standard deviation of the SAM dataset values.

    The image does not go through kornia's own antialiased :func:`~kornia.geometry.transform.resize`, which blurs with a
    Gaussian of standard deviation :math:`(f - 1) / 2` before downscaling by :math:`f`, a weaker low-pass than the
    antialiased bilinear filter of SAM's preprocessing. On a 1380 x 2076 image, against a transcription of Meta's
    ``SamPredictor`` driving the same MobileSAM weights, the three masks of a point prompt reach IoU 0.962, 0.967 and
    0.956 with kornia's filter and 0.969, 0.969 and 0.968 with SAM's.

    Two details depart on purpose from Meta's ``SamPredictor`` (``segment-anything@3518c86``):

    - The resized short side is truncated, where the reference rounds it (a 1380 x 2076 image becomes 680 x 1024
      here and 681 x 1024 there). Truncation is the rule of :class:`kornia.augmentation.Resize`, and the masks are
      resized back through the same recorded input size.
    - Prompts follow the half-pixel map :math:`x \mapsto (x + 0.5)\,s - 0.5`, kornia's pixel-centre convention and the
      map the image resize samples on, where the reference's ``apply_coords`` uses :math:`x \mapsto x\,s`. With the
      scale :math:`s = 8`, the keypoint (40, 30) goes to (323.5, 243.5) here and to (320, 240) in the reference. As
      in the reference's ``apply_boxes``, both corners of a box follow the point map: the box (10, 20, 60, 50) goes to
      (83.5, 163.5, 483.5, 403.5).

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

        Resize the image to the size given by ``transforms``, with SAM's filter (see the class docstring), and prepare
        it with the preprocess method.

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

        # The Resize augmentation gives the output size and the half-pixel matrix that maps the prompts; the image
        # itself is resized with the filter of SAM's reference (see the class docstring).
        self._tfs_params = self.transforms.forward_parameters(image.shape)
        resize_params = self._tfs_params[0].data
        if not isinstance(resize_params, dict):
            raise TypeError(f"Expected the parameters of the Resize augmentation. Got {type(resize_params)}.")
        height, width = (int(side) for side in resize_params["output_size"][0])
        image = _resize_like_sam(image, (height, width))
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

    def _valid_boxes(self, boxes: Boxes | torch.Tensor) -> torch.Tensor:
        """Validate the box prompts and return them as a ``(K, 4)`` xyxy tensor.

        The boxes are the ``K`` box prompts of the one image set. A ``(K, 4)`` or ``(1, K, 4)`` xyxy tensor is taken
        as given. A :class:`~kornia.geometry.boxes.Boxes` holding ``(K, 4, 2)`` or ``(1, K, 4, 2)`` vertices is read
        through ``to_tensor(mode="xyxy")``.
        """
        if isinstance(boxes, torch.Tensor):
            if boxes.ndim == 2:
                KORNIA_CHECK_SHAPE(boxes.data, ["K", "4"])
            else:
                KORNIA_CHECK_SHAPE(boxes.data, ["B", "K", "4"])
            # ``Boxes.from_tensor`` checks the values (finite, positive extents). Its vertices are not used: their
            # round trip widens a sub-pixel box ([10, 20, 10.5, 20.5] comes back as [9.5, 19.5, 11, 21]) and, in
            # bfloat16, rounds the -1 offset away ([256, 256, 258, 258] comes back as [256, 256, 256, 256]).
            Boxes.from_tensor(boxes, mode="xyxy")
            images = boxes.shape[0] if boxes.ndim == 3 else 1
            xyxy = boxes if boxes.is_floating_point() else boxes.to(torch.get_default_dtype())
        else:
            images = boxes.data.shape[0] if boxes.data.ndim == 4 else 1
            exported = boxes.to_tensor(mode="xyxy")
            if not isinstance(exported, torch.Tensor):
                raise TypeError(f"Expected the boxes as a tensor. Got {type(exported)}.")
            xyxy = exported

        KORNIA_CHECK(
            images == 1,
            f"VisualPrompter takes the boxes of its one image, (K, 4) or (1, K, 4); got boxes for {images} images. "
            "Use `Sam.forward` to prompt a batch of images.",
        )

        return xyxy.reshape(-1, 4)

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
        """Validate and preprocess the given prompts to be aligned with the input image.

        Keypoints and both corners of each xyxy box go through the half-pixel point map of ``transforms``. A box is
        mapped as its two corner points, as the reference's ``apply_boxes`` does, not through the vertices of
        :class:`~kornia.geometry.boxes.Boxes`, which place the far corner at ``x2 - 1``. A tensor box is taken as
        given; a :class:`~kornia.geometry.boxes.Boxes` is read through ``to_tensor(mode="xyxy")``.
        """
        points: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        if isinstance(keypoints, (Keypoints, torch.Tensor)) and isinstance(keypoints_labels, torch.Tensor):
            keypoints = self._valid_keypoints(keypoints, keypoints_labels)
            points = (self._transform_points(keypoints), keypoints_labels)
        elif keypoints is not None:
            KORNIA_CHECK(
                False,
                "keypoints_labels must be given with keypoints: a (K, N) tensor, 1 for a foreground point and 0 for "
                "a background point.",
            )

        bbox: Optional[torch.Tensor] = None
        if isinstance(boxes, (Boxes, torch.Tensor)):
            # (K, 4) xyxy -> K pairs of corner points -> the K box prompts, (K, 4).
            corners = self._valid_boxes(boxes).reshape(-1, 2, 2)
            bbox = self._transform_points(Keypoints.from_tensor(corners)).reshape(-1, 4)

        if isinstance(masks, torch.Tensor):
            self._valid_masks(masks)

        return Prompts(points=points, boxes=bbox, masks=masks)

    def _transform_points(self, keypoints: Keypoints) -> torch.Tensor:
        """Map points of the original image to the resized image, with the transform recorded by ``set_image``."""
        transformed = self._transform_prompts(keypoints, data_keys=["keypoints"])["keypoints"]
        points = transformed.to_tensor() if isinstance(transformed, Keypoints) else transformed
        if not isinstance(points, torch.Tensor):
            raise TypeError(f"Expected the transformed keypoints as a tensor. Got {type(points)}.")
        return points

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
                   :math:`(1, K, 4)`, whose corners are taken as given, or a :class:`~kornia.geometry.boxes.Boxes` of
                   shape :math:`(K, 4, 2)` or :math:`(1, K, 4, 2)`, read through ``to_tensor(mode="xyxy")``. `K` is
                   the number of prompts.
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
