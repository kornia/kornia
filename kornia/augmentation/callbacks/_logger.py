from typing import Any, Dict, List, Optional, Union

from kornia.augmentation.container.ops import DataType
from kornia.augmentation.container.params import ParamItem
from kornia.constants import DataKey

from kornia.core import Tensor, Module
from .base import AugmentationCallbackBase


class Logger(AugmentationCallbackBase):
    """Generic logging module.
    """
