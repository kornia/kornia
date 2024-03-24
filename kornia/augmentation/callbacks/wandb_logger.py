import importlib
from typing import Dict, List, Optional, Union

from kornia.augmentation.container.ops import DataType
from kornia.augmentation.container.params import ParamItem
from kornia.constants import DataKey
from kornia.core import Module, Tensor

from .base import AugmentationCallbackBase


class WandbLogger(AugmentationCallbackBase):
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
        run: Optional["wandb.Run"] = None,
        batches_to_log: int = -1,
        num_to_log: int = 4,
        log_indices: Optional[List[int]] = None,
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
        preprocessing: Optional[List[Optional[Module]]] = None,
    ):
        super().__init__()
        self.batches_to_log = batches_to_log
        self.log_indices = log_indices
        self.data_keys = data_keys
        self.preprocessing = preprocessing
        self.num_to_log = num_to_log
        if run is None:
            self.wandb = importlib.import_module("wandb")
        else:
            self.wandb = run

    def _make_mask_data(self, mask: Tensor):
        raise NotImplementedError

    def _make_bbox_data(self, mask: Tensor):
        raise NotImplementedError

    def on_sequential_forward_end(
        self,
        *args: Union[DataType, Dict[str, DataType]],
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ):
        """Called when `forward` ends for `AugmentationSequential`."""
        image_data = None
        mask_data = []
        box_data = []
        for i, (arg, data_key) in enumerate(zip(args, data_keys)):
            if i not in self.log_indices:
                continue

            preproc = self.preprocessing[self.log_indices[i]]
            out_arg = arg[: self.num_to_log]
            if preproc is not None:
                out_arg = preproc(out_arg)
            if data_key in [DataKey.INPUT]:
                image_data = out_arg
            if data_key in [DataKey.MASK]:
                mask_data = self._make_mask_data(out_arg)
            if data_key in [DataKey.BBOX, DataKey.BBOX_XYWH, DataKey.BBOX_XYXY]:
                box_data = self._make_bbox_data(out_arg)

        for i, (img, mask, box) in enumerate(zip(image_data, mask_data, box_data)):
            wandb_img = self.wandb.Image(img, masks=mask, boxes=box)
            self.wandb.log({"kornia_augmentation": wandb_img})
