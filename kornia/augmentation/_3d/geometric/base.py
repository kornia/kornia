from typing import Any, Dict, Optional

from kornia.augmentation._3d.base import RigidAffineAugmentationBase3D
from kornia.core import Tensor, as_tensor


class GeometricAugmentationBase3D(RigidAffineAugmentationBase3D):
    def get_transformation_matrix(
        self, input: Tensor, params: Optional[Dict[str, Tensor]] = None, flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        """Obtain transformation matrices.

        Return the current transformation matrix if existed. Generate a new one, otherwise.
        """
        flags = self.flags if flags is None else flags
        if params is not None and "transform_matrix" in params:
            transform = params["transform_matrix"]
        elif params is not None:
            transform = self.generate_transformation_matrix(input, params, flags)
        else:
            params = self.forward_parameters(input.shape)
            transform = params["transform_matrix"]
        return as_tensor(transform, device=input.device, dtype=input.dtype)
