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
from typing import Literal, Optional
import moviepy.editor as mpy
import torch

from kornia.models.depth_anything_3.architecture.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from kornia.models.depth_anything_3.specs import Prediction
from kornia.models.depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from kornia.models.depth_anything_3.utils.layout_helpers import hcat, vcat
from kornia.models.depth_anything_3.utils.visualize import vis_depth_map_tensor

VIDEO_QUALITY_MAP = {
    "low": {"crf": "28", "preset": "veryfast"},
    "medium": {"crf": "23", "preset": "medium"},
    "high": {"crf": "18", "preset": "slow"},
}


def export_to_gs_ply(
    prediction: Prediction,
    export_dir: str,
    gs_views_interval: Optional[
        int
    ] = 1,  # export GS every N views, useful for extremely dense inputs
):
    gs_world = prediction.gaussians
    pred_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)  # v h w 1
    idx = 0
    os.makedirs(os.path.join(export_dir, "gs_ply"), exist_ok=True)
    save_path = os.path.join(export_dir, f"gs_ply/{idx:04d}.ply")
    if gs_views_interval is None:  # select around 12 views in total
        gs_views_interval = max(pred_depth.shape[0] // 12, 1)
    save_gaussian_ply(
        gaussians=gs_world,
        save_path=save_path,
        ctx_depth=pred_depth,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=gs_views_interval,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )


def export_to_gs_video(
    prediction: Prediction,
    export_dir: str,
    extrinsics: Optional[torch.Tensor] = None,  # render views' world2cam, "b v 4 4"
    intrinsics: Optional[torch.Tensor] = None,  # render views' unnormed intrinsics, "b v 3 3"
    out_image_hw: Optional[tuple[int, int]] = None,  # render views' resolution, (h, w)
    chunk_size: Optional[int] = 4,
    trj_mode: Literal[
        "original",
        "smooth",
        "interpolate",
        "interpolate_smooth",
        "wander",
        "dolly_zoom",
        "extend",
        "wobble_inter",
    ] = "extend",
    color_mode: Literal["RGB+D", "RGB+ED"] = "RGB+ED",
    vis_depth: Optional[Literal["hcat", "vcat"]] = "hcat",
    enable_tqdm: Optional[bool] = True,
    output_name: Optional[str] = None,
    video_quality: Literal["low", "medium", "high"] = "high",
) -> None:
    gs_world = prediction.gaussians
    # if target poses are not provided, render the (smooth/interpolate) input poses
    if extrinsics is not None:
        tgt_extrs = extrinsics
    else:
        tgt_extrs = torch.from_numpy(prediction.extrinsics).unsqueeze(0).to(gs_world.means)
        if prediction.is_metric:
            scale_factor = prediction.scale_factor
            if scale_factor is not None:
                tgt_extrs[:, :, :3, 3] /= scale_factor
    tgt_intrs = (
        intrinsics
        if intrinsics is not None
        else torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(gs_world.means)
    )
    # if render resolution is not provided, render the input ones
    if out_image_hw is not None:
        H, W = out_image_hw
    else:
        H, W = prediction.depth.shape[-2:]
    # if single views, render wander trj
    if tgt_extrs.shape[1] <= 1:
        trj_mode = "wander"
        # trj_mode = "dolly_zoom"

    color, depth = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(H, W),
        chunk_size=chunk_size,
        trj_mode=trj_mode,
        use_sh=True,
        color_mode=color_mode,
        enable_tqdm=enable_tqdm,
    )

    # save as video
    ffmpeg_params = [
        "-crf",
        VIDEO_QUALITY_MAP[video_quality]["crf"],
        "-preset",
        VIDEO_QUALITY_MAP[video_quality]["preset"],
        "-pix_fmt",
        "yuv420p",
    ]  # best compatibility

    os.makedirs(os.path.join(export_dir, "gs_video"), exist_ok=True)
    for idx in range(color.shape[0]):
        video_i = color[idx]
        if vis_depth is not None:
            depth_i = vis_depth_map_tensor(depth[0])
            cat_fn = hcat if vis_depth == "hcat" else vcat
            video_i = torch.stack([cat_fn(c, d) for c, d in zip(video_i, depth_i)])
        frames = list(
            (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # T x H x W x C, uint8, numpy()

        fps = 24
        clip = mpy.ImageSequenceClip(frames, fps=fps)
        output_name = f"{idx:04d}_{trj_mode}" if output_name is None else output_name
        save_path = os.path.join(export_dir, f"gs_video/{output_name}.mp4")
        # clip.write_videofile(save_path, codec="libx264", audio=False, bitrate="4000k")
        clip.write_videofile(
            save_path,
            codec="libx264",
            audio=False,
            fps=fps,
            ffmpeg_params=ffmpeg_params,
        )
    return
