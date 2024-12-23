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

from .classification import ClassificationHead
from .connected_components import connected_components
from .diamond_square import diamond_square
from .distance_transform import DistanceTransform, distance_transform
from .edge_detection import EdgeDetector
from .extract_patches import (
    CombineTensorPatches,
    ExtractTensorPatches,
    combine_tensor_patches,
    compute_padding,
    extract_tensor_patches,
)
from .face_detection import *
from .histogram_matching import histogram_matching, interp
from .image_stitching import ImageStitcher
from .kmeans import KMeans
from .lambda_module import Lambda
from .models.tiny_vit import TinyViT
from .object_detection import ObjectDetector
from .vit import VisionTransformer
from .vit_mobile import MobileViT

__all__ = [
    "ClassificationHead",
    "CombineTensorPatches",
    "DistanceTransform",
    "EdgeDetector",
    "ExtractTensorPatches",
    "ImageStitcher",
    "KMeans",
    "Lambda",
    "MobileViT",
    "ObjectDetector",
    "TinyViT",
    "VisionTransformer",
    "combine_tensor_patches",
    "compute_padding",
    "connected_components",
    "diamond_square",
    "distance_transform",
    "extract_tensor_patches",
    "histogram_matching",
    "interp",
]
