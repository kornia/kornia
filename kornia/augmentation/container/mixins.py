from typing import Any, List, Optional, Tuple

from kornia.augmentation.callbacks import AugmentationCallbackBase
from kornia.core import Module, Tensor

__all__ = [
    "CallbacksMixIn",
    "TransformMatrixMinIn",
]


class CallbacksMixIn:
    """Enables callbacks life cycle."""
    def __init__(self, *args, **kwargs) -> None:  # type:ignore
        super().__init__(*args, **kwargs)
        self._callbacks: List[AugmentationCallbackBase] = []

    def register_callbacks(self, callbacks: AugmentationCallbackBase) -> None:
        [self._callbacks.append(cb) for cb in callbacks]

    def run_callbacks(self, hook: str, *args, **kwargs) -> None:
        for cb in self._callbacks:
            if not hasattr(cb, hook):
                continue

            hook_callable = getattr(cb, hook)

            if not callable(hook_callable):
                continue

            hook_callable(*args, **kwargs)


class TransformMatrixMinIn:
    """Enables computation matrix computation."""

    _valid_ops_for_transform_computation: Tuple[Any, ...] = ()
    _transformation_matrix_arg: str = "silent"

    def __init__(self, *args, **kwargs) -> None:  # type:ignore
        super().__init__(*args, **kwargs)
        self._transform_matrix: Optional[Tensor] = None
        self._transform_matrices: List[Optional[Tensor]] = []

    def _parse_transformation_matrix_mode(self, transformation_matrix_mode: str) -> None:
        _valid_transformation_matrix_args = {"silence", "silent", "rigid", "skip"}
        if transformation_matrix_mode not in _valid_transformation_matrix_args:
            raise ValueError(
                f"`transformation_matrix` has to be one of {_valid_transformation_matrix_args}. "
                f"Got {transformation_matrix_mode}."
            )
        self._transformation_matrix_arg = transformation_matrix_mode

    @property
    def transform_matrix(self) -> Optional[Tensor]:
        # In AugmentationSequential, the parent class is accessed first.
        # So that it was None in the beginning. We hereby use lazy computation here.
        if self._transform_matrix is None and len(self._transform_matrices) != 0:
            self._transform_matrix = self._transform_matrices[0]
            for mat in self._transform_matrices[1:]:
                self._update_transform_matrix(mat)
        return self._transform_matrix

    def _update_transform_matrix_for_valid_op(self, module: Module) -> None:
        raise NotImplementedError(module)

    def _update_transform_matrix_by_module(self, module: Module) -> None:
        if self._transformation_matrix_arg == "skip":
            return
        if isinstance(module, self._valid_ops_for_transform_computation):
            self._update_transform_matrix_for_valid_op(module)
        elif self._transformation_matrix_arg == "rigid":
            raise RuntimeError(
                f"Non-rigid module `{module}` is not supported under `rigid` computation mode. "
                "Please either update the module or change the `transformation_matrix` argument."
            )

    def _update_transform_matrix(self, transform_matrix: Optional[Tensor]) -> None:
        if self._transform_matrix is None:
            self._transform_matrix = transform_matrix
        else:
            self._transform_matrix = transform_matrix @ self._transform_matrix

    def _reset_transform_matrix_state(self) -> None:
        self._transform_matrix = None
        self._transform_matrices = []
