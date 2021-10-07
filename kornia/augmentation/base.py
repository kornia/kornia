import warnings
from typing import Any, Callable, List, cast, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

import kornia
from kornia.constants import DataKey
from kornia.utils.helpers import _torch_inverse_cast

from .utils import (
    _adapted_sampling,
    _transform_input,
    _transform_input3d,
    _transform_output_shape,
    _validate_input_dtype,
)

TensorWithTransformMat = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class _BasicAugmentationBase(nn.Module):
    r"""_BasicAugmentationBase base class for customized augmentation implementations.

    Plain augmentation base class without the functionality of transformation matrix calculations.
    By default, the random computations will be happened on CPU with ``torch.get_default_dtype()``.
    To change this behaviour, please use ``set_rng_device_and_dtype``.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities element-wise.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to
          the batch form ``False``.
    """

    def __init__(
        self, p: float = 0.5, p_batch: float = 1.0, same_on_batch: bool = False, keepdim: bool = False
    ) -> None:
        super().__init__()
        self.p = p
        self.p_batch = p_batch
        self.same_on_batch = same_on_batch
        self.keepdim = keepdim
        self._params: Dict[str, torch.Tensor] = {}
        if p != 0.0 or p != 1.0:
            self._p_gen = Bernoulli(self.p)
        if p_batch != 0.0 or p_batch != 1.0:
            self._p_batch_gen = Bernoulli(self.p_batch)
        self.set_rng_device_and_dtype(torch.device('cpu'), torch.get_default_dtype())

    def __repr__(self) -> str:
        return f"p={self.p}, p_batch={self.p_batch}, same_on_batch={self.same_on_batch}"

    def __unpack_input__(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def __check_batching__(self, input: TensorWithTransformMat):
        """Check if a transformation matrix is returned, it has to be in the same batching mode as output."""
        raise NotImplementedError

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return {}

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def set_rng_device_and_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.
        """
        self.device = device
        self.dtype = dtype

    def __batch_prob_generator__(
        self, batch_shape: torch.Size, p: float, p_batch: float, same_on_batch: bool
    ) -> torch.Tensor:
        batch_prob: torch.Tensor
        if p_batch == 1:
            batch_prob = torch.tensor([True])
        elif p_batch == 0:
            batch_prob = torch.tensor([False])
        else:
            batch_prob = _adapted_sampling((1,), self._p_batch_gen, same_on_batch).bool()

        if batch_prob.sum().item() == 1:
            elem_prob: torch.Tensor
            if p == 1:
                elem_prob = torch.tensor([True] * batch_shape[0])
            elif p == 0:
                elem_prob = torch.tensor([False] * batch_shape[0])
            else:
                elem_prob = _adapted_sampling((batch_shape[0],), self._p_gen, same_on_batch).bool()
            batch_prob = batch_prob * elem_prob
        else:
            batch_prob = batch_prob.repeat(batch_shape[0])
        return batch_prob

    def forward_parameters(self, batch_shape) -> Dict[str, torch.Tensor]:
        to_apply = self.__batch_prob_generator__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        _params = self.generate_parameters(torch.Size((int(to_apply.sum().item()), *batch_shape[1:])))
        if _params is None:
            _params = {}
        _params['batch_prob'] = to_apply
        return _params

    def apply_func(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> TensorWithTransformMat:
        input = self.transform_tensor(input)
        return self.apply_transform(input, params)

    def forward(  # type: ignore
        self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None  # type: ignore
    ) -> TensorWithTransformMat:
        in_tensor = self.__unpack_input__(input)
        self.__check_batching__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.forward_parameters(batch_shape)
        self._params = params

        output = self.apply_func(input, self._params)
        return _transform_output_shape(output, ori_shape) if self.keepdim else output


class _AugmentationBase(_BasicAugmentationBase):
    r"""_AugmentationBase base class for customized augmentation implementations.

    Advanced augmentation base class with the functionality of transformation matrix calculations.

    Args:
        pprobability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        return_transform: if ``True`` return the matrix describing the geometric transformation applied to each
          input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def __init__(
        self,
        return_transform: bool = None,
        same_on_batch: bool = False,
        p: float = 0.5,
        p_batch: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.p = p
        self.p_batch = p_batch
        self.return_transform = return_transform

    def __repr__(self) -> str:
        return super().__repr__() + f", return_transform={self.return_transform}"

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def __unpack_input__(  # type: ignore
        self, input: TensorWithTransformMat
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(input, tuple):
            in_tensor = input[0]
            in_transformation = input[1]
            return in_tensor, in_transformation
        in_tensor = input
        return in_tensor, None

    def apply_func(  # type: ignore
        self,
        in_tensor: torch.Tensor,
        in_transform: Optional[torch.Tensor],  # type: ignore
        params: Dict[str, torch.Tensor],
        return_transform: bool = False,
    ) -> TensorWithTransformMat:
        to_apply = params['batch_prob']

        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            output = in_tensor
            trans_matrix = self.identity_matrix(in_tensor)
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            trans_matrix = self.compute_transformation(in_tensor, params)
            output = self.apply_transform(in_tensor, params, trans_matrix)
        else:
            output = in_tensor.clone()
            trans_matrix = self.identity_matrix(in_tensor)
            trans_matrix[to_apply] = self.compute_transformation(in_tensor[to_apply], params)
            output[to_apply] = self.apply_transform(in_tensor[to_apply], params, trans_matrix[to_apply])

        self._transform_matrix = trans_matrix

        if return_transform:
            out_transformation = trans_matrix if in_transform is None else trans_matrix @ in_transform
            return output, out_transformation

        if in_transform is not None:
            return output, in_transform

        return output

    def forward(  # type: ignore
        self,
        input: TensorWithTransformMat,
        params: Optional[Dict[str, torch.Tensor]] = None,  # type: ignore
        return_transform: Optional[bool] = None,
    ) -> TensorWithTransformMat:
        in_tensor, in_transform = self.__unpack_input__(input)
        self.__check_batching__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape

        if return_transform is None:
            return_transform = self.return_transform
        return_transform = cast(bool, return_transform)
        if params is None:
            params = self.forward_parameters(batch_shape)
        if 'batch_prob' not in params:
            params['batch_prob'] = torch.tensor([True] * batch_shape[0])
            warnings.warn("`batch_prob` is not found in params. Will assume applying on all data.")

        self._params = params
        output = self.apply_func(in_tensor, in_transform, self._params, return_transform)
        return _transform_output_shape(output, ori_shape) if self.keepdim else output


class AugmentationBase2D(_AugmentationBase):
    r"""AugmentationBase2D base class for customized augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        return_transform: if ``True`` return the matrix describing the geometric transformation applied to each
          input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.
    """

    def __check_batching__(self, input: TensorWithTransformMat):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 4:
                if len(mat.shape) != 3:
                    raise AssertionError('Input tensor is in batch mode ' 'but transformation matrix is not')
                if mat.shape[0] != inp.shape[0]:
                    raise AssertionError(
                        f"In batch dimension, input has {inp.shape[0]} but transformation matrix has {mat.shape[0]}"
                    )
            elif len(inp.shape) in (2, 3):
                if len(mat.shape) != 2:
                    raise AssertionError("Input tensor is in non-batch mode but transformation matrix is not")
            else:
                raise ValueError(f'Unrecognized output shape. Expected 2, 3, or 4, got {len(inp.shape)}')

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def identity_matrix(self, input) -> torch.Tensor:
        """Return 3x3 identity matrix."""
        return kornia.eye_like(3, input)


class IntensityAugmentationBase2D(AugmentationBase2D):
    r"""IntensityAugmentationBase2D base class for customized intensity augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        return_transform: if ``True`` return the matrix describing the geometric transformation applied to each
          input tensor. If ``False`` and the input is a tuple the applied transformation  won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)


class GeometricAugmentationBase2D(AugmentationBase2D):
    r"""GeometricAugmentationBase2D base class for customized geometric augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        return_transform: if ``True`` return the matrix describing the geometric transformation applied to each
          input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def inverse_transform(
        self,
        input: torch.Tensor,
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """By default, the exact transformation as ``apply_transform`` will be used."""
        raise NotImplementedError

    def compute_inverse_transformation(self, transform: torch.Tensor):
        """Compute the inverse transform of given transformation matrices."""
        return _torch_inverse_cast(transform)

    def get_transformation_matrix(
        self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if params is not None:
            transform = self.compute_transformation(input, params)
        elif not hasattr(self, "_transform_matrix"):
            params = self.forward_parameters(input.shape)
            transform = self.identity_matrix(input)
            transform[params['batch_prob']] = self.compute_transformation(input[params['batch_prob']], params)
        else:
            transform = self._transform_matrix
        return torch.as_tensor(transform, device=input.device, dtype=input.dtype)

    def inverse(
        self,
        input: TensorWithTransformMat,
        params: Optional[Dict[str, torch.Tensor]] = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(input, (list, tuple)):
            input, transform = input
        else:
            transform = self.get_transformation_matrix(input, params)
        if params is not None:
            transform = self.identity_matrix(input)
            transform[params['batch_prob']] = self.compute_transformation(input[params['batch_prob']], params)

        ori_shape = input.shape
        in_tensor = self.transform_tensor(input)
        batch_shape = input.shape
        if params is None:
            params = self._params
        if size is None and "input_size" in params:
            # Majorly for cropping functions
            size = params['input_size'].unique(dim=0).squeeze().numpy().tolist()
            size = (size[0], size[1])
        if 'batch_prob' not in params:
            params['batch_prob'] = torch.tensor([True] * batch_shape[0])
            warnings.warn("`batch_prob` is not found in params. Will assume applying on all data.")
        output = input.clone()
        to_apply = params['batch_prob']
        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            output = in_tensor
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            transform = self.compute_inverse_transformation(transform)
            output = self.inverse_transform(in_tensor, transform, size, **kwargs)
        else:
            transform[to_apply] = self.compute_inverse_transformation(transform[to_apply])
            output[to_apply] = self.inverse_transform(in_tensor[to_apply], transform[to_apply], size, **kwargs)
        return cast(torch.Tensor, _transform_output_shape(output, ori_shape)) if self.keepdim else output


class AugmentationBase3D(_AugmentationBase):
    r"""AugmentationBase3D base class for customized augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        return_transform: if ``True`` return the matrix describing the geometric transformation applied to each
          input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
    """

    def __check_batching__(self, input: TensorWithTransformMat):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 5:
                if len(mat.shape) != 3:
                    raise AssertionError('Input tensor is in batch mode ' 'but transformation matrix is not')
                if mat.shape[0] != inp.shape[0]:
                    raise AssertionError(
                        f"In batch dimension, input has {inp.shape[0]} but transformation matrix has {mat.shape[0]}"
                    )
            elif len(inp.shape) in (3, 4):
                if len(mat.shape) != 2:
                    raise AssertionError("Input tensor is in non-batch mode but transformation matrix is not")
            else:
                raise ValueError(f'Unrecognized output shape. Expected 3, 4 or 5, got {len(inp.shape)}')

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input3d(input)

    def identity_matrix(self, input) -> torch.Tensor:
        """Return 4x4 identity matrix."""
        return kornia.eye_like(4, input)


class MixAugmentationBase(_BasicAugmentationBase):
    r"""MixAugmentationBase base class for customized mix augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required.
    "apply_transform" will need to handle the probabilities internally.

    Args:
        p: probability for applying an augmentation. This param controls if to apply the augmentation for the batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def __init__(self, p: float, p_batch: float, same_on_batch: bool = False, keepdim: bool = False) -> None:
        super().__init__(p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)

    def __check_batching__(self, input: TensorWithTransformMat):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 4:
                if len(mat.shape) != 3:
                    raise AssertionError('Input tensor is in batch mode ' 'but transformation matrix is not')
                if mat.shape[0] != inp.shape[0]:
                    raise AssertionError(
                        f"In batch dimension, input has {inp.shape[0]} but transformation matrix has {mat.shape[0]}"
                    )
            elif len(inp.shape) in (2, 3):
                if len(mat.shape) != 2:
                    raise AssertionError("Input tensor is in non-batch mode but transformation matrix is not")
            else:
                raise ValueError(f'Unrecognized output shape. Expected 2, 3, or 4, got {len(inp.shape)}')

    def __unpack_input__(  # type: ignore
        self, input: TensorWithTransformMat
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(input, tuple):
            in_tensor = input[0]
            in_transformation = input[1]
            return in_tensor, in_transformation
        in_tensor = input
        return in_tensor, None

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def apply_transform(  # type: ignore
        self, input: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def apply_func(  # type: ignore
        self, in_tensor: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        to_apply = params['batch_prob']

        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            output = in_tensor
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            output, label = self.apply_transform(in_tensor, label, params)
        else:
            raise ValueError(
                "Mix augmentations must be performed batch-wisely. Element-wise augmentation is not supported."
            )

        return output, label

    def forward(  # type: ignore
        self,
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[TensorWithTransformMat, torch.Tensor]:
        in_tensor, in_trans = self.__unpack_input__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        # If label is not provided, it would output the indices instead.
        if label is None:
            if isinstance(input, (tuple, list)):
                device = input[0].device
            else:
                device = input.device
            label = torch.arange(0, in_tensor.size(0), device=device, dtype=torch.long)
        if params is None:
            batch_shape = in_tensor.shape
            params = self.forward_parameters(batch_shape)
        self._params = params

        output, lab = self.apply_func(in_tensor, label, self._params)
        output = _transform_output_shape(output, ori_shape) if self.keepdim else output  # type: ignore
        if in_trans is not None:
            return (output, in_trans), lab
        return output, lab


class LambdaAugmentation(_BasicAugmentationBase):
    """Lambda augmentation for wrapping general image operations.

    This function supports chaining general image operations in an ``AugmentationSequential`` container.
    The ``LambdaAugmentation`` assumes no randomness happened across a batch of images (an LambdaAugmentation
    will apply to all images in a batch).

    Args:
        input: arbitary transformation function for an image tensor,
            with a shape of :math:`(B, C, H, W)`.
        mask: arbitary transformation function for a mask tensor,
            with a shape of :math:`(B, C, H, W)`.
        bbox: arbitary transformation function for a bounding box tensor,
            with a shape of :math:`(B, N, 4, 2)`.
        keypoints: arbitary transformation function for a keypoint tensor,
            with a shape of :math:`(B, N, 2)`.
        input_inverse: arbitary inverse transformation function for an image tensor,
            with a shape of :math:`(B, C, H, W)`.
        mask_inverse: arbitary inverse transformation function for a mask tensor,
            with a shape of :math:`(B, C, H, W)`.
        bbox_inverse: arbitary inverse transformation function for a bounding box tensor,
            with a shape of :math:`(B, N, 4, 2)`.
        keypoints_inverse: arbitary inverse transformation function for a keypoint tensor,
            with a shape of :math:`(B, N, 2)`.
        param_generator: generate sharable parameter across functions. It should accept an
            input shape tensor, and returns a dict of parameters. If not given, no shared parameters
            will be generated.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    .. code-block:: python

        # Despite of implementing a child class by inheriting this method.
        # Another option is to use the provided `from_modules` class contructor
        # to construct a simple augmentation from a pre-defined operation.

        PatchExtractor = LambdaAugmentation.from_modules(
            kornia.contrib.ExtractTensorPatches(),
            inverse_module=kornia.contrib.CombineTensorPatches(),
            for_data_keys=["input", "mask"],
        )
    """
    def __init__(
        self,
        input: Optional[Callable] = None,
        mask: Optional[Callable] = None,
        bbox: Optional[Callable] = None,
        keypoints: Optional[Callable] = None,
        input_inverse: Optional[Callable] = None,
        mask_inverse: Optional[Callable] = None,
        bbox_inverse: Optional[Callable] = None,
        keypoints_inverse: Optional[Callable] = None,
        param_generator: Optional[Callable] = None,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=1., p_batch=1.0, same_on_batch=True, keepdim=keepdim)
        self.input = input
        self.mask = mask
        self.bbox = bbox
        self.keypoints = keypoints
        self.input_inverse = input_inverse
        self.mask_inverse = mask_inverse
        self.bbox_inverse = bbox_inverse
        self.keypoints_inverse = keypoints_inverse
        self.param_generator = param_generator

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def __check_batching__(self, input: TensorWithTransformMat):
        pass

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        return _transform_input(input)

    @classmethod
    def from_modules(
        cls,
        forward_module: nn.Module,
        inverse_module: Optional[nn.Module] = None,
        for_data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
        default_fn: str = "none"
    ) -> "LambdaAugmentation":
        """Construct a lambda augmentation from nn.Modules.

        Args:
            forward_module: required. A module to apply the transformation.
            inverse_module: optional. A module to inverse the transformation.
            for_data_keys: determine which data keys will consume the forward and inverse module.
            default_fn: the expected default behaviour for not defined data keys, "none" | "return_identity".
                If "none", a ``NotImplementedError`` will be raised.
                If "return_identity", the exact input tensor will be returned.
        """
        for_data_keys = [DataKey.get(inp) for inp in for_data_keys]
        default_module: Optional[nn.Module] = None
        if default_fn == "none":
            default_module = None
        elif default_fn == "return_identity":
            default_module = nn.Identity()
        else:
            raise NotImplementedError

        return cls(
            input=forward_module if DataKey.INPUT in for_data_keys else default_module,
            input_inverse=inverse_module if DataKey.INPUT in for_data_keys else default_module,
            mask=forward_module if DataKey.MASK in for_data_keys else default_module,
            mask_inverse=inverse_module if DataKey.MASK in for_data_keys else default_module,
            bbox=forward_module if DataKey.BBOX in for_data_keys else default_module,
            bbox_inverse=inverse_module if DataKey.BBOX in for_data_keys else default_module,
            keypoints=forward_module if DataKey.KEYPOINTS in for_data_keys else default_module,
            keypoints_inverse=inverse_module if DataKey.KEYPOINTS in for_data_keys else default_module,
            param_generator=None,
            keepdim=False,
        )

    def __unpack_input__(  # type: ignore
        self, input: TensorWithTransformMat
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(input, tuple):
            in_tensor = input[0]
            in_transformation = input[1]
            return in_tensor, in_transformation
        in_tensor = input
        return in_tensor, None

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        """Generate shared parameters for applying the same transformation on different data format."""
        if self.param_generator is not None:
            return self.param_generator(batch_shape)
        return super().generate_parameters(batch_shape)

    def forward_input(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the transformation on input shaped as :match:`(B, C, H, W)`."""
        if self.input is not None:
            return self.input(input, **kwargs)
        raise NotImplementedError

    def inverse_input(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Inverse the transformation on input shaped as :match:`(B, C, H, W)`."""
        if self.input_inverse is not None:
            return self.input_inverse(input, **kwargs)
        raise NotImplementedError

    def forward_mask(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the transformation on mask shaped as :match:`(B, C, H, W)`."""
        if self.mask is not None:
            return self.mask(input, **kwargs)
        raise NotImplementedError

    def inverse_mask(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Inverse the transformation on mask shaped as :match:`(B, C, H, W)`."""
        if self.mask_inverse is not None:
            return self.mask_inverse(input, **kwargs)
        raise NotImplementedError

    def forward_bbox(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the transformation on boxes shaped as :match:`(B, N, 4, 2)`."""
        if self.bbox is not None:
            return self.bbox(input, **kwargs)
        raise NotImplementedError

    def inverse_bbox(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Inverse the transformation on boxes shaped as :match:`(B, N, 4, 2)`."""
        if self.bbox_inverse is not None:
            return self.bbox_inverse(input, **kwargs)
        raise NotImplementedError

    def forward_keypoints(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the transformation on keypoints shaped as :match:`(B, N, 2)`."""
        if self.keypoints is not None:
            return self.keypoints(input, **kwargs)
        raise NotImplementedError

    def inverse_keypoints(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Inverse the transformation on keypoints shaped as :match:`(B, N, 2)`."""
        if self.keypoints_inverse is not None:
            return self.keypoints_inverse(input, **kwargs)
        raise NotImplementedError

    def forward_by_key(
        self,
        input: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]],
        data_key: Union[str, int, DataKey],
    ) -> torch.Tensor:
        """Apply a transformation given its data key."""
        if params is None:
            params = {}
        if DataKey.get(data_key) == DataKey.INPUT:
            return cast(torch.Tensor, self.forward(input, params))
        if DataKey.get(data_key) == DataKey.MASK:
            return self.forward_mask(input, **params)
        if DataKey.get(data_key) == DataKey.BBOX:
            return self.forward_bbox(input, **params)
        if DataKey.get(data_key) == DataKey.BBOX_XYHW:
            # Convert to polygon format then convert back
            from_shape = input.shape
            input = input.view(-1, input.size(-1))
            boxes = kornia.geometry.bbox_generator(input[..., 0], input[..., 1], input[..., 2], input[..., 3])
            boxes = self.forward_bbox(boxes, **params)
            x, y = boxes[:, 0, 0], boxes[:, 0, 1]
            w = boxes[:, 1, 0] - boxes[:, 0, 0] + 1
            h = boxes[:, 2, 1] - boxes[:, 0, 1] + 1
            boxes = torch.stack([x, y, w, h]).view(from_shape)
            return boxes
        if DataKey.get(data_key) == DataKey.BBOX_XYXY:
            # Convert to polygon format then convert back
            from_shape = input.shape
            input = input.view(-1, input.size(-1))
            boxes = kornia.geometry.bbox_generator(
                input[..., 0], input[..., 1], input[..., 2] - input[..., 0], input[..., 3] - input[..., 1])
            boxes = self.forward_bbox(boxes, **params)
            x, y = boxes[:, 0, 0], boxes[:, 0, 1]
            x2 = boxes[:, 1, 0] + 1
            y2 = boxes[:, 2, 1] + 1
            boxes = torch.stack([x, y, x2, y2]).view(from_shape)
            return boxes
        if DataKey.get(data_key) == DataKey.KEYPOINTS:
            return self.forward_keypoints(input, **params)
        raise NotImplementedError

    def inverse_by_key(
        self,
        input: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]],
        data_key: Union[str, int, DataKey],
    ) -> torch.Tensor:
        """Inverse a transformation given its data key."""
        if params is None:
            params = self._params
        if DataKey.get(data_key) == DataKey.INPUT:
            return self.inverse(input, params)
        if DataKey.get(data_key) == DataKey.MASK:
            return self.inverse_mask(input, **params)
        if DataKey.get(data_key) == DataKey.BBOX:
            return self.inverse_bbox(input, **params)
        if DataKey.get(data_key) == DataKey.BBOX_XYHW:
            # Convert to polygon format then convert back
            # TODO: wrapped it in kornia.bbox
            from_shape = input.shape
            input = input.view(-1, input.size(-1))
            boxes = kornia.geometry.bbox_generator(input[..., 0], input[..., 1], input[..., 2], input[..., 3])
            boxes = self.inverse_bbox(boxes, **params)
            x, y = boxes[:, 0, 0], boxes[:, 0, 1]
            w = boxes[:, 1, 0] - boxes[:, 0, 0] + 1
            h = boxes[:, 2, 1] - boxes[:, 0, 1] + 1
            boxes = torch.stack([x, y, w, h]).view(from_shape)
            return boxes
        if DataKey.get(data_key) == DataKey.BBOX_XYXY:
            # Convert to polygon format then convert back
            from_shape = input.shape
            input = input.view(-1, input.size(-1))
            boxes = kornia.geometry.bbox_generator(
                input[..., 0], input[..., 1], input[..., 2] - input[..., 0], input[..., 3] - input[..., 1])
            boxes = self.inverse_bbox(boxes, **params)
            x, y = boxes[:, 0, 0], boxes[:, 0, 1]
            x2 = boxes[:, 1, 0] + 1
            y2 = boxes[:, 2, 1] + 1
            boxes = torch.stack([x, y, x2, y2]).view(from_shape)
            return boxes
        if DataKey.get(data_key) == DataKey.KEYPOINTS:
            return self.inverse_keypoints(input, **params)
        raise NotImplementedError

    def inverse(self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if params is None:
            params = self._params
        return self.inverse_input(input, **params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_input(input, **params)

    def forward(  # type: ignore
        self,
        input: TensorWithTransformMat,
        params: Optional[Dict[str, torch.Tensor]] = None,  # type: ignore
    ) -> TensorWithTransformMat:
        in_tensor, in_transform = self.__unpack_input__(input)
        self.__check_batching__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape

        if params is None:
            params = self.forward_parameters(batch_shape)
            # Remove ``batch_prob``
            if "batch_prob" in params:
                del params["batch_prob"]

        self._params = params
        output = self.apply_func(in_tensor, self._params)
        output = _transform_output_shape(output, ori_shape) if self.keepdim else output
        if in_transform is None:
            return output
        return cast(torch.Tensor, output), in_transform
