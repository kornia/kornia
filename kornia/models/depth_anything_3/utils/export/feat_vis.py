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

import os
import cv2
import imageio
import numpy as np
from tqdm.auto import tqdm

from kornia.models.depth_anything_3.utils.parallel_utils import async_call
from kornia.models.depth_anything_3.utils.pca_utils import PCARGBVisualizer


@async_call
def export_to_feat_vis(
    prediction,
    export_dir,
    fps=15,
):
    """Export feature visualization with PCA.

    Args:
        prediction: Model prediction containing feature maps
        export_dir: Directory to export results
        fps: Frame rate for output video (default: 15)
    """
    out_dir = os.path.join(export_dir, "feat_vis")
    os.makedirs(out_dir, exist_ok=True)

    images = prediction.processed_images
    for k, v in prediction.aux.items():
        if not k.startswith("feat_layer_"):
            continue
        os.makedirs(os.path.join(out_dir, k), exist_ok=True)
        viz = PCARGBVisualizer(basis_mode="fixed", percentile_mode="global", clip_percent=10.0)
        viz.fit_reference(v)
        feats_vis = viz.transform_video(v)
        for idx in tqdm(range(len(feats_vis))):
            img = images[idx]
            feat_vis = (feats_vis[idx] * 255).astype(np.uint8)
            feat_vis = cv2.resize(
                feat_vis, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            save_path = os.path.join(out_dir, f"{k}/{idx:06d}.jpg")
            save = np.concatenate([img, feat_vis], axis=1)
            imageio.imwrite(save_path, save, quality=95)
        cmd = (
            "ffmpeg -loglevel error -hide_banner -y "
            f"-framerate {fps} -start_number 0 "
            f"-i {out_dir}/{k}/%06d.jpg "
            f"-c:v libx264 -pix_fmt yuv420p "
            f"{out_dir}/{k}.mp4"
        )
        os.system(cmd)
