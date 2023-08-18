from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.augmentation.utils import _validate_input_dtype
from kornia.constants import DataKey, DType
from kornia.core import Tensor, tensor

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
        data_keys: the input type sequential for applying augmentations.
          Accepts "input", "mask".

    Note:
        - This augmentation requires that segmentation masks are available for all images in the batch and that at
          least some objects in the image are annotated.
        - This implementation works for arbitrary spatial dimensions including 2D and 3D images.

    Inputs:
        - Input image tensors: :math:`(B, C, *)`.
        - Segmentation mask tensors: :math:`(B, *)`.
        - (optional) Additional image or mask tensors with are transformed in the same ways as the input image:
          :math:`(B, C, *)` or :math:`(B, *)`.

    Returns:
        Union[Tuple[Tensor, Tensor], List[Tensor]]:

        Tuple[Tensor, Tensor]:
            - Augmented image tensors: :math:`(B, C, *)`.
            - Augmented mask tensors: :math:`(B, *)`.
        List[Tensor]:
            - Augmented image tensors: :math:`(B, C, *)`.
            - Augmented mask tensors: :math:`(B, *)`.
            - Additional augmented image or mask tensors: :math:`(B, C, *)` or :math:`(B, *)`.

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

    You can apply the same augmentation again, but the new mask should contain the same labels for the augmentation
    to have an effect:

        >>> aug._params["selected_labels"]  # Image 0 received label 2 from image 1 and image 1 label 0 from image 0
        [tensor(2), tensor(0)]
        >>> mask2 = torch.randint(0, 3, (2, 5, 5))
        >>> image_out2, mask_out2 = aug(image, mask2, params=aug._params)
        >>> mask_out2
        tensor([[[2, 2, 2, 0, 1],
                 [1, 2, 0, 1, 1],
                 [0, 0, 1, 2, 2],
                 [0, 0, 1, 2, 1],
                 [1, 1, 1, 2, 0]],
        <BLANKLINE>
                [[0, 2, 2, 0, 0],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 2, 0],
                 [0, 0, 1, 0, 1],
                 [1, 1, 1, 2, 0]]])
    """

    def __init__(
        self,
        excluded_labels: Optional[Sequence[int]] = None,
        p: float = 0.5,
        p_batch: float = 1.0,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch)

        if excluded_labels is None:
            excluded_labels = []
        self.excluded_labels = tensor(excluded_labels)
        if self.excluded_labels.ndim != 1:
            raise ValueError(
                f"excluded_labels must be a 1-dimensional sequence, but got {self.excluded_labels.ndim} dimensions."
            )

        if data_keys is None:
            data_keys = [DataKey.INPUT, DataKey.MASK]
        self.data_keys = [DataKey.get(inp) for inp in data_keys]
        self._channel_dim = 1

    def apply_non_transform_mask(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return input

    def transform_input(self, acceptor: Tensor, donor: Tensor, selection: Tensor) -> Tensor:
        # Expand selection to the channel dimension
        selection = selection.unsqueeze(dim=self._channel_dim).expand_as(donor)
        acceptor[selection] = donor[selection]
        return acceptor

    def transform_mask(self, acceptor: Tensor, donor: Tensor, selection: Tensor) -> Tensor:
        acceptor[selection] = donor[selection]
        return acceptor

    def forward(  # type: ignore[override]
        self,
        image: Tensor,
        mask: Tensor,
        *additional_inputs: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[Tuple[Tensor, Tensor], List[Tensor]]:
        keys: List[DataKey]
        if data_keys is None:
            keys = self.data_keys
        else:
            keys = [DataKey.get(inp) for inp in data_keys]

        inputs: List[Tensor] = [image, mask, *additional_inputs]
        if len(keys) != len(inputs):
            raise RuntimeError(f"Length of keys ({len(keys)}) does not match number of inputs ({len(inputs)}).")
        if keys[:2] != [DataKey.INPUT, DataKey.MASK]:
            raise RuntimeError(
                f"The first two keys must be {DataKey.INPUT} (image) and {DataKey.MASK} (segmentation mask),"
                f" but got {keys[:2]}."
            )

        _validate_input_dtype(image, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

        if image.ndim != mask.ndim + 1:
            raise RuntimeError(
                "image must have one additional dimension (channel dimension) than mask,"
                f" but got {image.ndim} and {mask.ndim}."
            )

        if mask.size() != torch.Size([s for i, s in enumerate(image.size()) if i != self._channel_dim]):
            raise RuntimeError(
                "The dimensions of image and mask must match except for the channel dimension),"
                f" but got {mask.size()} and {image.size()}."
            )

        if params is None:
            self._params = self.forward_parameters(image.shape)
            self._params.update({"dtype": tensor(DType.get(image.dtype).value)})
        else:
            self._params = params

        if self.excluded_labels.device != mask.device:
            self.excluded_labels = self.excluded_labels.to(mask.device)

        # Image below are the donors
        acceptor_indices = torch.where(self._params["batch_prob"] > 0.5)[0]
        donor_indices = (acceptor_indices - 1) % len(self._params["batch_prob"])
        selection = torch.zeros((len(acceptor_indices), *mask.shape[1:]), dtype=torch.bool, device=mask.device)

        if "selected_labels" not in self._params:
            selected_labels: List[Tensor] = []
            for d in range(len(donor_indices)):
                # Select a random label from the donor image
                current_mask = mask[donor_indices[d]]
                labels = current_mask.unique()

                # Remove any label which is part of the excluded labels
                labels = labels[(labels.view(1, -1) != self.excluded_labels.view(-1, 1)).all(dim=0)]

                if len(labels) > 0:
                    selected_label = labels[torch.randperm(len(labels))[0]]

                    selection[d].masked_fill_(current_mask == selected_label, True)
                    selected_labels.append(selected_label)

            self._params["selected_labels"] = selected_labels
        else:
            selected_labels = self._params["selected_labels"]
            if len(selected_labels) != len(acceptor_indices):
                raise RuntimeError(
                    f"Length of selected_labels ({len(selected_labels)}) in the parameters does not match the number"
                    f" of images where this augmentation should be applied ({len(acceptor_indices)})."
                )

            for d, selected_label in zip(range(len(donor_indices)), selected_labels):
                current_mask = mask[donor_indices[d]]
                selection[d].masked_fill_(current_mask == selected_label, True)

        outputs: List[Tensor] = []
        for dcate, _input in zip(keys, inputs):
            acceptor = _input[acceptor_indices].clone()
            donor = _input[donor_indices]

            output: Tensor
            if dcate == DataKey.INPUT:
                applied = self.transform_input(acceptor, donor, selection)
                output = self.apply_non_transform(_input, params, self.flags)
                output = output.index_put(
                    (acceptor_indices,), self.apply_non_transform_mask(applied, params, self.flags)
                )
            elif dcate == DataKey.MASK:
                applied = self.transform_mask(acceptor, donor, selection)
                output = self.apply_non_transform_mask(_input, params, self.flags)
                output = output.index_put(
                    (acceptor_indices,), self.apply_non_transform_mask(applied, params, self.flags)
                )
            else:
                raise NotImplementedError

            outputs.append(output)

        if len(outputs) == 2:
            return outputs[0], outputs[1]
        else:
            return outputs
