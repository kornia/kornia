# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Record selections and parameter draws made by sequential augmentation containers.

Which record to use:

- ``ParamItem`` stores the child name and its parameter data.
- ``PatchParamItem`` adds the patch indices for a patch-level operation.
"""

from typing import Dict, List, NamedTuple, Optional, Union

import torch

__all__ = ["ParamItem", "PatchParamItem"]


class ParamItem(NamedTuple):
    """Record the selected child and sampled parameters of one sequential operation.

    Sequential containers create ``ParamItem(name, data)`` records in ``_params``; passing that list back through
    ``params=`` replays the recorded selection and parameters. Use ``PatchParamItem`` when the record also needs patch
    indices.

    Attributes:
        name: The name of the augmentation operation.
        data: The dictionary of parameters or a list of nested parameters.

    See Also:
        - :class:`~kornia.augmentation.container.ImageSequential`: records these items when it runs children.

    """

    name: str
    data: Optional[Union[Dict[str, torch.Tensor], List["ParamItem"]]]


class PatchParamItem(NamedTuple):
    """Record the patch indices and parameters for one patch-sequence operation.

    :class:`~kornia.augmentation.container.PatchSequential` uses ``PatchParamItem(indices, param)`` to replay a
    child on the same patch locations. Use ``ParamItem`` for a non-patch sequence.

    Attributes:
        indices: The list of patch indices where the augmentation is applied.
        param: The specific :class:`~kornia.augmentation.container.params.ParamItem` containing the operation
        parameters.

    See Also:
        - :class:`~kornia.augmentation.container.PatchSequential`: creates these patch-specific records.

    """

    indices: List[int]
    param: ParamItem
