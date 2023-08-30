from typing import Any, Dict, Optional

from torch import Tensor

from kornia.augmentation._pc.base import RigidAffineAugmentationBasePC
from kornia.geometry.boxes import Boxes3D
from kornia.geometry.keypoints import Keypoints3D


class IntensityAugmentationBasePC(RigidAffineAugmentationBasePC):
    r"""IntensityAugmentationBasePC base class for customized intensity augmentation implementations.

    Intensity-based point cloud augmentation methods are the transformations without using a specific
    transformation matrix.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    def apply_non_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # For the images where batch_prob == False.
        return input

    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_non_transform_boxes(
        self, input: Boxes3D, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes3D:
        return input

    def apply_transform_boxes(
        self, input: Boxes3D, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes3D:
        return input

    def apply_non_transform_keypoint(
        self, input: Keypoints3D, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints3D:
        return input

    def apply_transform_keypoint(
        self, input: Keypoints3D, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints3D:
        return input

    def apply_non_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input
