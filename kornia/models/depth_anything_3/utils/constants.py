# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_EXPORT_DIR = "workspace/gallery/scene"
DEFAULT_GALLERY_DIR = "workspace/gallery"
DEFAULT_GRADIO_DIR = "workspace/gradio"
THRESH_FOR_REF_SELECTION = 3

# =============================================================================
# Benchmark Evaluation Constants
# =============================================================================

# Default evaluation workspace directory
DEFAULT_EVAL_WORKSPACE = "workspace/evaluation"

# Default reference view selection strategy for evaluation
# Use "first" for consistent and reproducible evaluation results
# Other options: "saddle_balanced", "auto", "mid"
EVAL_REF_VIEW_STRATEGY = "first"

# -----------------------------------------------------------------------------
# DTU Dataset Configuration
# Reference: https://roboimagedata.compute.dtu.dk/
# Note: DepthAnything3 was never trained on any images from DTU.
# -----------------------------------------------------------------------------

# Root directory for DTU evaluation data (MVSNet format)
# Download from: https://drive.google.com/file/d/1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D/view
DTU_EVAL_DATA_ROOT = "workspace/benchmark_dataset/dtu"

# List of DTU evaluation scenes
DTU_SCENES = [
    "scan1",
    "scan4",
    "scan9",
    "scan10",
    "scan11",
    "scan12",
    "scan13",
    "scan15",
    "scan23",
    "scan24",
    "scan29",
    "scan32",
    "scan33",
    "scan34",
    "scan48",
    "scan49",
    "scan62",
    "scan75",
    "scan77",
    "scan110",
    "scan114",
    "scan118",
]

# Point cloud fusion hyperparameters
DTU_DIST_THRESH = 0.2  # Distance threshold for geometric consistency (mm)
DTU_NUM_CONSIST = 4  # Minimum number of consistent views for a point
DTU_MAX_POINTS = 4_000_000  # Maximum points in fused point cloud

# 3D reconstruction evaluation hyperparameters
DTU_DOWN_DENSE = 0.2  # Downsample density for evaluation (mm)
DTU_PATCH_SIZE = 60  # Patch size for boundary handling
DTU_MAX_DIST = 20  # Outlier threshold for accuracy/completeness (mm)

# -----------------------------------------------------------------------------
# DTU-64 Dataset Configuration (Pose Evaluation Only)
# This is a subset of DTU with 64 images per scene for pose evaluation.
# Note: This dataset is ONLY for pose evaluation, not 3D reconstruction.
# -----------------------------------------------------------------------------

# Root directory for DTU-64 evaluation data
DTU64_EVAL_DATA_ROOT = "workspace/benchmark_dataset/dtu64"
DTU64_CAMERA_ROOT = "workspace/benchmark_dataset/dtu64/Cameras"

# List of DTU-64 evaluation scenes (13 scenes)
DTU64_SCENES = [
    "scan105",
    "scan114",
    "scan118",
    "scan122",
    "scan24",
    "scan37",
    "scan40",
    "scan55",
    "scan63",
    "scan65",
    "scan69",
    "scan83",
    "scan97",
]

# -----------------------------------------------------------------------------
# ETH3D Dataset Configuration
# Reference: https://www.eth3d.net/
# High-resolution multi-view stereo benchmark with laser-scanned ground truth.
# Note: DepthAnything3 was never trained on any images from ETH3D.
# -----------------------------------------------------------------------------

# Root directory for ETH3D evaluation data
ETH3D_EVAL_DATA_ROOT = "workspace/benchmark_dataset/eth3d"

# List of ETH3D evaluation scenes (indoor and outdoor)
ETH3D_SCENES = [
    "courtyard",
    "electro",
    "kicker",
    "pipes",
    "relief",
    # "terrace",  # Excluded: known issues
    "delivery_area",
    "facade",
    # "meadow",   # Excluded: known issues
    "office",
    "playground",
    "relief_2",
    "terrains",
]

# Images to filter out (known problematic views per scene)
ETH3D_FILTER_KEYS = {
    "delivery_area": ["711.JPG", "712.JPG", "713.JPG", "714.JPG"],
    "electro": ["9289.JPG", "9290.JPG", "9291.JPG", "9292.JPG", "9293.JPG", "9298.JPG"],
    "playground": ["587.JPG", "588.JPG", "589.JPG", "590.JPG", "591.JPG", "592.JPG"],
    "relief": [
        "427.JPG", "428.JPG", "429.JPG", "430.JPG", "431.JPG", "432.JPG",
        "433.JPG", "434.JPG", "435.JPG", "436.JPG", "437.JPG", "438.JPG",
    ],
    "relief_2": [
        "458.JPG", "459.JPG", "460.JPG", "461.JPG", "462.JPG", "463.JPG",
        "464.JPG", "465.JPG", "466.JPG", "467.JPG", "468.JPG",
    ],
}

# TSDF fusion hyperparameters (scaled for outdoor scenes)
ETH3D_VOXEL_LENGTH = 4.0 / 512.0 * 5  # Voxel size for TSDF (meters)
ETH3D_SDF_TRUNC = 0.04 * 5  # SDF truncation distance (meters)
ETH3D_MAX_DEPTH = 100000.0  # Maximum depth for integration (effectively no truncation)

# Point cloud sampling
ETH3D_SAMPLING_NUMBER = 1_000_000  # Number of points to sample from mesh

# 3D reconstruction evaluation hyperparameters
ETH3D_EVAL_THRESHOLD = 0.05 * 5  # Distance threshold for precision/recall (meters)
ETH3D_DOWN_SAMPLE = 4.0 / 512.0 * 5  # Voxel size for evaluation downsampling (meters)


# ==============================================================================
# 7Scenes Dataset Configuration
# ==============================================================================
# Reference: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
# Note: Indoor RGB-D dataset with ground truth poses and meshes.

# Root directory for 7Scenes evaluation data
SEVENSCENES_EVAL_DATA_ROOT = "workspace/benchmark_dataset/7scenes"

# List of 7Scenes evaluation scenes
SEVENSCENES_SCENES = [
    "chess",
    "fire",
    "heads",
    "office",
    "pumpkin",
    "redkitchen",
    "stairs",
]

# Fixed camera intrinsics for 7Scenes (all images share same intrinsics)
SEVENSCENES_FX = 585.0
SEVENSCENES_FY = 585.0
SEVENSCENES_CX = 320.0
SEVENSCENES_CY = 240.0

# TSDF fusion hyperparameters (indoor scenes, smaller voxels)
SEVENSCENES_VOXEL_LENGTH = 4.0 / 512.0  # Voxel size for TSDF (meters)
SEVENSCENES_SDF_TRUNC = 0.04  # SDF truncation distance (meters)
SEVENSCENES_MAX_DEPTH = 1000000.0  # Maximum depth for integration (no truncation)

# Point cloud sampling
SEVENSCENES_SAMPLING_NUMBER = 1_000_000  # Number of points to sample from mesh

# 3D reconstruction evaluation hyperparameters
SEVENSCENES_EVAL_THRESHOLD = 0.05  # Distance threshold for precision/recall (meters)
SEVENSCENES_DOWN_SAMPLE = 4.0 / 512.0  # Voxel size for evaluation downsampling (meters)


# ==============================================================================
# ScanNet++ Dataset Configuration
# ==============================================================================
# Reference: https://kaldir.vc.in.tum.de/scannetpp/
# Note: High-quality indoor RGB-D dataset with iPhone and DSLR images.

# Root directory for ScanNet++ evaluation data
SCANNETPP_EVAL_DATA_ROOT = "workspace/benchmark_dataset/scannetpp"

# List of ScanNet++ evaluation scenes
SCANNETPP_SCENES = [
    "09c1414f1b",
    "1ada7a0617",
    "40aec5fffa",
    "3e8bba0176",
    "acd95847c5",
    "578511c8a9",
    "5f99900f09",
    "c4c04e6d6c",
    "f3d64c30f8",
    "7bc286c1b6",
    "c5439f4607",
    "286b55a2bf",
    "fb5a96b1a2",
    "7831862f02",
    "38d58a7a31",
    "bde1e479ad",
    "9071e139d9",
    "21d970d8de",
    "bcd2436daf",
    "cc5237fd77",
]

# Input resolution for ScanNet++ (after undistortion and resize)
SCANNETPP_INPUT_H = 768
SCANNETPP_INPUT_W = 1024

# TSDF fusion hyperparameters (indoor scenes)
SCANNETPP_VOXEL_LENGTH = 0.02  # Voxel size for TSDF (meters)
SCANNETPP_SDF_TRUNC = 0.15  # SDF truncation distance (meters)
SCANNETPP_MAX_DEPTH = 5.0  # Maximum depth for integration (meters)

# Point cloud sampling
SCANNETPP_SAMPLING_NUMBER = 1_000_000  # Number of points to sample from mesh

# 3D reconstruction evaluation hyperparameters
SCANNETPP_EVAL_THRESHOLD = 0.05  # Distance threshold for precision/recall (meters)
SCANNETPP_DOWN_SAMPLE = 0.02  # Voxel size for evaluation downsampling (meters)


# ==============================================================================
# HiRoom Dataset Configuration
# ==============================================================================
# Note: Indoor RGB-D dataset.

# Root directory for HiRoom evaluation data
HIROOM_EVAL_DATA_ROOT = "workspace/benchmark_dataset/hiroom/data"
HIROOM_GT_ROOT_PATH = "workspace/benchmark_dataset/hiroom/fused_pcd"
HIROOM_SCENE_LIST_PATH = "workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt"

# TSDF fusion hyperparameters (indoor scenes)
HIROOM_VOXEL_LENGTH = 4.0 / 512.0  # Voxel size for TSDF (meters)
HIROOM_SDF_TRUNC = 0.04  # SDF truncation distance (meters)
HIROOM_MAX_DEPTH = 10000.0  # Maximum depth for integration (no truncation)

# Point cloud sampling
HIROOM_SAMPLING_NUMBER = 1_000_000  # Number of points to sample from mesh

# 3D reconstruction evaluation hyperparameters
HIROOM_EVAL_THRESHOLD = 0.05  # Distance threshold for precision/recall (meters)
HIROOM_DOWN_SAMPLE = 4.0 / 512.0  # Voxel size for evaluation downsampling (meters)
