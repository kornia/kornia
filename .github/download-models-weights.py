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

import argparse
import logging
import os

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Models used in doctests and tests
# Format: "name": url
# All models use torch.hub.load_state_dict_from_url
MODELS = {
    # SOLD2 - line detection
    "sold2_wireframe": "http://cmp.felk.cvut.cz/~mishkdmy/models/sold2_wireframe.pth",
    # DexiNed - edge detection (used by EdgeDetector doctest)
    "dexined": "http://cmp.felk.cvut.cz/~mishkdmy/models/DexiNed_BIPED_10.pth",
    # DISK - feature extraction (used by DISK.from_pretrained doctest)
    "disk_depth": "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth",
    # DeDoDe - feature detection/description (used by DeDoDe.from_pretrained doctest)
    "dedode_detector_L_v2": "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth",
    "dedode_descriptor_B_SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_descriptor_setting_C.pth",
    "dedode_descriptor_B_upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth",
    "dedode_descriptor_G_upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth",
    # DINOv2 - used by DeDoDe encoder
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    # YuNet - face detection
    "yunet": "https://github.com/kornia/data/raw/main/yunet_final.pth",
    # RT-DETR - object detection (used by RTDETR.from_pretrained doctest)
    "rtdetr_r18vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
    # VisionTransformer - used by VisionTransformer.from_config doctest
    "vit_b_16": "https://huggingface.co/kornia/vit_b16_augreg_i21k_r224/resolve/main/vit_b-16.pth",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WeightsDownloader")
    parser.add_argument("--target_directory", "-t", required=False, default="weights")

    args = parser.parse_args()

    # Set torch.hub directory - files will go to {target_directory}/checkpoints/
    torch.hub.set_dir(args.target_directory)
    # For HuggingFace model caching
    os.environ["HF_HOME"] = args.target_directory

    logger.info(f"Downloading models to: {torch.hub.get_dir()}/checkpoints/")

    for name, url in MODELS.items():
        logger.info(f"Downloading `{name}` from `{url}`...")
        # Don't pass model_dir - use the default from torch.hub.set_dir()
        # This ensures files go to {hub_dir}/checkpoints/ matching test behavior
        torch.hub.load_state_dict_from_url(url, map_location=torch.device("cpu"))

    logger.info("All models downloaded successfully!")
    raise SystemExit(0)
