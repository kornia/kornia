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

from __future__ import annotations

from typing import Union

import torch

# classes
Tensor = torch.Tensor
tensor = torch.tensor
Module = torch.nn.Module
ModuleList = torch.nn.ModuleList
Parameter = torch.nn.Parameter
Sequential = torch.nn.Sequential

# functions
# NOTE: ideally we expose what we find in numpy
arange = torch.arange
concatenate = torch.cat
stack = torch.stack
linspace = torch.linspace
normalize = torch.nn.functional.normalize
pad = torch.nn.functional.pad
eye = torch.eye
einsum = torch.einsum
zeros = torch.zeros
zeros_like = torch.zeros_like
ones = torch.ones
ones_like = torch.ones_like
where = torch.where
complex = torch.complex
diag = torch.diag
softmax = torch.nn.functional.softmax
map_coordinates = torch.nn.functional.grid_sample
cos = torch.cos
sin = torch.sin
tan = torch.tan
deg2rad = torch.deg2rad
rad2deg = torch.rad2deg

# constructors
as_tensor = torch.as_tensor
from_numpy = torch.from_numpy

# random
rand = torch.rand

# type alias
Device = Union[str, torch.device, None]
Dtype = Union[torch.dtype, None]
