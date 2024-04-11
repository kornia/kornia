from typing import List, Union

from kornia.core import Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

DataType = Union[Tensor, List[Tensor], Boxes, Keypoints]

# NOTE: shouldn't this SequenceDataType alias be equals to List[DataType]?
SequenceDataType = Union[List[Tensor], List[List[Tensor]], List[Boxes], List[Keypoints]]
