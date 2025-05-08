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

# NOTE: kornia filters and geometry must go first since are the core of the library
# and by changing the import order you might get into a circular dependencies issue.
from . import filters
from . import geometry
from . import grad_estimator

# import the other modules for convenience
from . import (
    augmentation,
    color,
    contrib,
    core,
    config,
    enhance,
    feature,
    io,
    losses,
    metrics,
    models,
    morphology,
    onnx,
    tracking,
    utils,
    x,
)

# Multi-framework support using ivy
from .transpiler import to_jax, to_numpy, to_tensorflow

# NOTE: we are going to expose to top level very few things
from kornia.constants import pi
from kornia.utils import (
    eye_like,
    vec_like,
    create_meshgrid,
    image_to_tensor,
    tensor_to_image,
    xla_is_available,
)

# Version variable
__version__ = "0.8.1"
