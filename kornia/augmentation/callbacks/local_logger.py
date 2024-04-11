import importlib
from typing import Dict, List, Optional, Union

from kornia.augmentation.container.ops import DataType, SequenceDataType
from kornia.augmentation.container.params import ParamItem
from kornia.constants import DataKey
from kornia.core import Module, Tensor

from .base import AugmentationCallback


class WandbLogger(AugmentationCallback):
    """Logging images onto W&B for `AugmentationSequential`.

    Args:
        batches_to_save: the number of batches to be logged. -1 is to save all batches.
        num_to_log: number of images to log in a batch.
        log_indices: only selected input types are logged. If `log_indices=[0, 2]` and
                     `data_keys=["input", "bbox", "mask"]`, only the images and masks
                     will be logged.
        data_keys: the input type sequential. Accepts "input", "image", "mask",
                   "bbox", "bbox_xyxy", "bbox_xywh", "keypoints".
        preprocessing: add preprocessing for images if needed. If not None, the length
                       must match `data_keys`.
    """

    def __init__(
        self,
        log_dir: str = "./kornia_logs",
        batches_to_save: int = 10,
        num_to_log: int = 4,
        log_indices: Optional[List[int]] = None,
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
        postprocessing: Optional[List[Optional[Module]]] = None,
    ):
        super().__init__(
            batches_to_log=batches_to_log,
            num_to_log=num_to_log,
            log_indices=log_indices,
            data_keys=data_keys,
            postprocessing=postprocessing,
        )
        self.log_dir = log_dir

    def _make_mask_data(self, mask: Tensor):
        ...

    def _make_bbox_data(self, bbox: Tensor):
        ...

    def _log_data(self, data: SequenceDataType):
        ...