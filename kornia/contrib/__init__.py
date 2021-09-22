from kornia.contrib.classification import ClassificationHead
from kornia.contrib.connected_components import connected_components
from kornia.contrib.extract_patches import extract_tensor_patches, ExtractTensorPatches
from kornia.contrib.vit import VisionTransformer
from kornia.contrib.extract_patches import (
    extract_tensor_patches,
    ExtractTensorPatches,
    combine_tensor_patches,
    CombineTensorPatches
)

__all__ = [
    "connected_components",
    "extract_tensor_patches",
    "ExtractTensorPatches",
    "combine_tensor_patches",
    "CombineTensorPatches",
    "VisionTransformer",
    "ClassificationHead"
]
