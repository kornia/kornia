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

"""Contributed image operations and task-level model wrappers.

Which function or class to use, with the module form after the function:

- Cut an image into sliding-window patches: :func:`extract_tensor_patches`, :class:`ExtractTensorPatches`; put them
  back, averaging overlaps: :func:`combine_tensor_patches`, :class:`CombineTensorPatches`; the padding that makes the
  windows fit: :func:`compute_padding`.
- Label 8-connected foreground components: by a fixed number of pooling steps, :func:`connected_components`;
  exactly, with no step budget, :func:`connected_components_union_find`.
- Euclidean distance to the nearest positive pixel, in 2D or 3D: :func:`distance_transform`,
  :class:`DistanceTransform`.
- Match the histogram of an image to a template: :func:`histogram_matching`; the 1-D linear interpolation it uses,
  which extrapolates the end segments: :func:`interp`.
- Plasma-fractal noise maps: :func:`diamond_square`.
- Cluster points by k-means: :class:`KMeans`.
- Run a Python callable as a module: :class:`Lambda`.
- Audit the geometry of one augmentation pipeline call: :func:`audit`, which returns an
  :class:`AugmentationAuditReport` of :class:`AugmentationAuditStep` and :class:`SpatialAudit` entries.
- Detect edges with a wrapped network: :class:`EdgeDetector`; build one around DexiNed, pretrained by default:
  :class:`EdgeDetectorBuilder`.
- Detect faces and their landmarks: :class:`FaceDetector`.
- Detect objects: :class:`ObjectDetector`, built by :class:`RTDETRDetectorBuilder`.
- Upscale images: :class:`SuperResolution`, configured by :class:`SuperResolutionConfig` and built by
  :class:`SmallSRBuilder` or :class:`RRDBNetBuilder`.
- Stitch images with overlapping fields of view: :class:`ImageStitcher`.
- Segment one image from point, box and mask prompts: :class:`~kornia.contrib.visual_prompter.VisualPrompter`.
- Classify images with the TinyViT network: :class:`TinyViT`.
"""

from kornia.models.tiny_vit import TinyViT

from .augmentation_audit import AugmentationAuditReport, AugmentationAuditStep, SpatialAudit, audit
from .connected_components import connected_components, connected_components_union_find
from .diamond_square import diamond_square
from .distance_transform import DistanceTransform, distance_transform
from .edge_detection import EdgeDetector, EdgeDetectorBuilder
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
from .object_detection import ObjectDetector, RTDETRDetectorBuilder
from .super_resolution import RRDBNetBuilder, SmallSRBuilder, SuperResolution, SuperResolutionConfig

__all__ = [
    "AugmentationAuditReport",
    "AugmentationAuditStep",
    "CombineTensorPatches",
    "DistanceTransform",
    "EdgeDetector",
    "EdgeDetectorBuilder",
    "ExtractTensorPatches",
    "ImageStitcher",
    "KMeans",
    "Lambda",
    "ObjectDetector",
    "RRDBNetBuilder",
    "RTDETRDetectorBuilder",
    "SmallSRBuilder",
    "SpatialAudit",
    "SuperResolution",
    "SuperResolutionConfig",
    "TinyViT",
    "audit",
    "combine_tensor_patches",
    "compute_padding",
    "connected_components",
    "connected_components_union_find",
    "diamond_square",
    "distance_transform",
    "extract_tensor_patches",
    "histogram_matching",
    "interp",
]
