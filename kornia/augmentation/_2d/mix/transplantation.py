from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch

from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.augmentation.utils import _validate_input_dtype
from kornia.constants import DataKey
from kornia.core import Tensor, tensor
from kornia.core.check import KORNIA_CHECK

__all__ = ["RandomTransplantation"]


class RandomTransplantation(MixAugmentationBaseV2):
    r"""RandomTransplantation augmentation.

    .. image:: _static/img/RandomTransplantation.png

    Randomly transplant (copy and paste) image features and corresponding segmentation masks between images in a batch.
    The transplantation transform works as follows:

        1. Based on the parameter `p`, a certain number of images in the batch are selected as acceptor of a
           transplantation.
        2. For each acceptor, the image below in the batch is selected as donor (via circling: :math:`i - 1 \mod B`).
        3. From the donor, a random label is selected and the corresponding image features and segmentation mask are
           transplanted to the acceptor.

    The augmentation is described in `Semantic segmentation of surgical hyperspectral images under geometric domain
    shifts` :cite:`sellner2023semantic`.

    Args:
        excluded_labels: sequence of labels which should not be transplanted from a donor. This can be useful if only
          parts of the image are annotated and the non-annotated regions (with a specific label index) should be
          excluded from the augmentation. If no label is left in the donor image, nothing is transplanted.
        p: probability for applying an augmentation to an image. This parameter controls how many images in a batch
          receive a transplant.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        data_keys: the input type sequential for applying augmentations. There must be at least one "mask" tensor. If no
          data keys are given, the first tensor is assumed to be `DataKey.INPUT` and the second tensor `DataKey.MASK`.
          Accepts "input", "mask".

    Note:
        - This augmentation requires that segmentation masks are available for all images in the batch and that at
          least some objects in the image are annotated.
        - When using this class directly (`RandomTransplantation()(...)`), it works for arbitrary spatial dimensions
          including 2D and 3D images. When wrapping in :class:`kornia.augmentation.AugmentationSequential`, use
          :class:`kornia.augmentation.RandomTransplantation` for 2D and
          :class:`kornia.augmentation.RandomTransplantation3D` for 3D images.

    Inputs:
        - Segmentation mask tensor which is used to determine the objects for transplantation: :math:`(B, *)`.
        - (optional) Additional image or mask tensors where the features are transplanted based on the first
          segmentation mask: :math:`(B, C, *)` (`DataKey.INPUT`) or :math:`(B, *)` (`DataKey.MASK`).

    Returns:
        Tensor | list[Tensor]:

        Tensor:
            - Augmented mask tensors: :math:`(B, *)`.
        list[Tensor]:
            - Augmented mask tensors: :math:`(B, *)`.
            - Additional augmented image or mask tensors: :math:`(B, C, *)` (`DataKey.INPUT`) or :math:`(B, *)`
              (`DataKey.MASK`).

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> aug = RandomTransplantation(p=1.)
        >>> image = torch.randn(2, 3, 5, 5)
        >>> mask = torch.randint(0, 3, (2, 5, 5))
        >>> mask
        tensor([[[0, 0, 1, 1, 0],
                 [1, 2, 0, 0, 0],
                 [1, 2, 1, 1, 0],
                 [0, 0, 0, 0, 2],
                 [2, 2, 2, 0, 2]],
        <BLANKLINE>
                [[2, 0, 0, 2, 1],
                 [2, 1, 0, 2, 1],
                 [2, 0, 1, 0, 2],
                 [2, 2, 2, 0, 2],
                 [2, 1, 0, 0, 0]]])
        >>> image_out, mask_out = aug(image, mask)
        >>> image_out.shape
        torch.Size([2, 3, 5, 5])
        >>> mask_out.shape
        torch.Size([2, 5, 5])
        >>> mask_out
        tensor([[[2, 0, 1, 2, 0],
                 [2, 2, 0, 2, 0],
                 [2, 2, 1, 1, 2],
                 [2, 2, 2, 0, 2],
                 [2, 2, 2, 0, 2]],
        <BLANKLINE>
                [[0, 0, 0, 2, 0],
                 [2, 1, 0, 0, 0],
                 [2, 0, 1, 0, 0],
                 [0, 0, 0, 0, 2],
                 [2, 1, 0, 0, 0]]])
        >>> aug._params["selected_labels"]  # Image 0 received label 2 from image 1 and image 1 label 0 from image 0
        tensor([2, 0])

    You can apply the same augmentation again in which case the same objects get transplanted between the images:

        >>> aug._params["selection"]  # The pixels (objects) which get transplanted
        tensor([[[ True, False, False,  True, False],
                 [ True, False, False,  True, False],
                 [ True, False, False, False,  True],
                 [ True,  True,  True, False,  True],
                 [ True, False, False, False, False]],
        <BLANKLINE>
                [[ True,  True, False, False,  True],
                 [False, False,  True,  True,  True],
                 [False, False, False, False,  True],
                 [ True,  True,  True,  True, False],
                 [False, False, False,  True, False]]])
        >>> image2 = torch.zeros(2, 3, 5, 5)
        >>> image2[1] = 1
        >>> image2[:, 0]
        tensor([[[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                [[1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1.]]])
        >>> image_out2, mask_out2 = aug(image2, mask, params=aug._params)
        >>> image_out2[:, 0]
        tensor([[[1., 0., 0., 1., 0.],
                 [1., 0., 0., 1., 0.],
                 [1., 0., 0., 0., 1.],
                 [1., 1., 1., 0., 1.],
                 [1., 0., 0., 0., 0.]],
        <BLANKLINE>
                [[0., 0., 1., 1., 0.],
                 [1., 1., 0., 0., 0.],
                 [1., 1., 1., 1., 0.],
                 [0., 0., 0., 0., 1.],
                 [1., 1., 1., 0., 1.]]])
    """

    def __init__(
        self,
        excluded_labels: Optional[Union[Sequence[int], Tensor]] = None,
        p: float = 0.5,
        p_batch: float = 1.0,
        data_keys: Optional[list[str | int | DataKey]] = None,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch)

        if excluded_labels is None:
            excluded_labels = []
        if not isinstance(excluded_labels, Tensor):
            excluded_labels = tensor(excluded_labels)
        self.excluded_labels: Tensor = excluded_labels
        KORNIA_CHECK(
            self.excluded_labels.ndim == 1,
            f"excluded_labels must be a 1-dimensional sequence, but got {self.excluded_labels.ndim} dimensions.",
        )

        if data_keys is None:
            data_keys = [DataKey.INPUT, DataKey.MASK]
        self.data_keys = [DataKey.get(inp) for inp in data_keys]
        self._channel_dim = 1

    def apply_non_transform_mask(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        return input

    def transform_input(self, acceptor: Tensor, donor: Tensor, selection: Tensor) -> Tensor:  # type: ignore[override]
        # Expand selection to the channel dimension
        selection = selection.unsqueeze(dim=self._channel_dim).expand_as(donor)
        acceptor[selection] = donor[selection]
        return acceptor

    def transform_mask(self, acceptor: Tensor, donor: Tensor, selection: Tensor) -> Tensor:  # type: ignore[override]
        acceptor[selection] = donor[selection]
        return acceptor

    def params_from_input(
        self,
        *input: Tensor,
        data_keys: list[DataKey],
        params: dict[str, Tensor],
        extra_args: Optional[dict[DataKey, dict[str, Any]]] = None,
    ) -> dict[str, Tensor]:
        """Compute parameters for the transformation which are based on one or more input tensors.

        This function is, for example, called by :class:`kornia.augmentation.container.ops.AugmentationSequentialOps`
        before the augmentation is applied on the individual input tensors.

        Args:
            *input: All input tensors passed to the augmentation pipeline.
            data_keys: Associated data key for every input tensor.
            params: Dictionary of parameters computed so far by the augmentation pipeline (e.g. including the
                    `batch_prob`).
            extra_args: Optional dictionary of extra arguments with specific options for different input types.

        Returns:
             Updated dictionary of parameters with the necessary information to apply the augmentation on all input
             tensors separately.
        """
        KORNIA_CHECK(
            len(data_keys) == len(input),
            f"Length of keys ({len(data_keys)}) does not match number of inputs ({len(input)}).",
        )

        # The first mask key will be used for the transplantation
        mask: Tensor = input[data_keys.index(DataKey.MASK)]
        for _input, key in zip(input, data_keys):
            if key == DataKey.INPUT:
                KORNIA_CHECK(
                    _input.ndim == mask.ndim + 1,
                    "Every image input must have one additional dimension (channel dimension) than the segmentation "
                    f"mask, but got {_input.ndim} for the input image and {mask.ndim} for the segmentation mask.",
                )
                KORNIA_CHECK(
                    mask.size() == torch.Size([s for i, s in enumerate(_input.size()) if i != self._channel_dim]),
                    "The dimensions of the input image and segmentation mask must match except for the channel "
                    f"dimension, but got {_input.size()} for the input image and {mask.size()} for the segmentation "
                    "mask.",
                )

        if "acceptor_indices" not in params:
            params["acceptor_indices"] = torch.where(params["batch_prob"] > 0.5)[0]
        if "donor_indices" not in params:
            params["donor_indices"] = (params["acceptor_indices"] - 1) % len(params["batch_prob"])

        if "selected_labels" not in params:
            if self.excluded_labels.device != mask.device:
                self.excluded_labels = self.excluded_labels.to(mask.device)

            donor_labels: list[Tensor] = []
            for d in range(len(params["donor_indices"])):
                # Select a random label from the donor image
                current_mask = mask[params["donor_indices"][d]]
                labels = current_mask.unique()

                # Remove any label which is part of the excluded labels
                labels = labels[(labels.view(1, -1) != self.excluded_labels.view(-1, 1)).all(dim=0)]

                if len(labels) > 0:
                    selected_label = labels[torch.randperm(len(labels))[0]]
                    donor_labels.append(selected_label)

            params["selected_labels"] = torch.stack(donor_labels) if len(donor_labels) > 0 else torch.empty(0)

        if "selection" not in params:
            selection = torch.zeros(
                (len(params["acceptor_indices"]), *mask.shape[1:]), dtype=torch.bool, device=mask.device
            )
            selected_labels: Tensor = params["selected_labels"]
            KORNIA_CHECK(
                selected_labels.ndim == 1,
                f"selected_labels must be a 1-dimensional tensor, but got {selected_labels.ndim} dimensions.",
            )
            KORNIA_CHECK(
                len(selected_labels) <= len(params["acceptor_indices"]),
                f"There cannot be more selected labels ({len(selected_labels)}) than images where this augmentation "
                f"should be applied ({len(params['acceptor_indices'])}).",
            )

            for d, selected_label in zip(range(len(params["donor_indices"])), selected_labels):
                current_mask = mask[params["donor_indices"][d]]
                selection[d].masked_fill_(current_mask == selected_label, True)

            params["selection"] = selection

        return params

    def forward(  # type: ignore[override]
        self,
        *input: Tensor,
        params: Optional[dict[str, Tensor]] = None,
        data_keys: Optional[list[str | int | DataKey]] = None,
        **kwargs: dict[str, Any],
    ) -> Tensor | list[Tensor]:
        keys: list[DataKey]
        if data_keys is None:
            keys = self.data_keys
        else:
            keys = [DataKey.get(inp) for inp in data_keys]

        if params is None:
            mask: Tensor = input[keys.index(DataKey.MASK)]
            self._params = self.forward_parameters(mask.shape)
        else:
            self._params = params

        if any(k not in self._params for k in ["acceptor_indices", "donor_indices", "selection"]):
            self._params.update(self.params_from_input(*input, data_keys=keys, params=self._params))

        outputs: list[Tensor] = []
        for dcate, _input in zip(keys, input):
            acceptor = _input[self._params["acceptor_indices"]].clone()
            donor = _input[self._params["donor_indices"]]

            output: Tensor
            if dcate == DataKey.INPUT:
                _validate_input_dtype(_input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

                applied = self.transform_input(acceptor, donor, self._params["selection"])
                output = self.apply_non_transform(_input, self._params, self.flags)
                output = output.index_put(
                    (self._params["acceptor_indices"],),
                    self.apply_non_transform_mask(applied, self._params, self.flags),
                )
            elif dcate == DataKey.MASK:
                applied = self.transform_mask(acceptor, donor, self._params["selection"])
                output = self.apply_non_transform_mask(_input, self._params, self.flags)
                output = output.index_put(
                    (self._params["acceptor_indices"],),
                    self.apply_non_transform_mask(applied, self._params, self.flags),
                )
            else:
                raise NotImplementedError

            outputs.append(output)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
