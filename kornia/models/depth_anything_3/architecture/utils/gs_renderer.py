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

import math
from math import isqrt
from typing import Literal, Optional

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from kornia.models.depth_anything_3.specs import Gaussians
from kornia.models.depth_anything_3.utils.camera_trj_helpers import (
    interpolate_extrinsics,
    interpolate_intrinsics,
    render_dolly_zoom_path,
    render_stabilization_path,
    render_wander_path,
    render_wobble_inter_path,
)
from kornia.models.depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, get_fov
from kornia.models.depth_anything_3.utils.logger import logger

try:
    from gsplat import rasterization
except ImportError:
    logger.warn(
        "Dependency `gsplat` is required for rendering 3DGS. "
        "Install via: pip install git+https://github.com/nerfstudio-project/"
        "gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"
    )


def render_3dgs(
    extrinsics: torch.Tensor,  # "batch_views 4 4", w2c
    intrinsics: torch.Tensor,  # "batch_views 3 3", normalized
    image_shape: tuple[int, int],
    gaussian: Gaussians,
    background_color: Optional[torch.Tensor] = None,  # "batch_views 3"
    use_sh: bool = True,
    num_view: int = 1,
    color_mode: Literal["RGB+D", "RGB+ED"] = "RGB+D",
    **kwargs,
) -> tuple[
    torch.Tensor,  # "batch_views 3 height width"
    torch.Tensor,  # "batch_views height width"
]:
    # extract gaussian params
    gaussian_means = gaussian.means
    gaussian_scales = gaussian.scales
    gaussian_quats = gaussian.rotations
    gaussian_opacities = gaussian.opacities
    gaussian_sh_coefficients = gaussian.harmonics
    b, _, _ = extrinsics.shape

    if background_color is None:
        background_color = repeat(torch.tensor([0.0, 0.0, 0.0]), "c -> b c", b=b).to(gaussian_sh_coefficients)

    if use_sh:
        _, _, _, n = gaussian_sh_coefficients.shape
        degree = isqrt(n) - 1
        shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()
    else:  # use color
        shs = gaussian_sh_coefficients.squeeze(-1).sigmoid().contiguous()  # (b, g, c), normed to (0, 1)

    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()
    focal_length_x = w / (2 * tan_fov_x)
    focal_length_y = h / (2 * tan_fov_y)

    view_matrix = extrinsics.float()

    all_images = []
    all_radii = []
    all_depths = []
    # render view in a batch based, each batch contains one scene
    # assume the Gaussian parameters are originally repeated along the view dim
    batch_scene = b // num_view

    def index_i_gs_attr(full_attr, idx):
        # return rearrange(full_attr, "(b v) ... -> b v ...", v=num_view)[idx, 0]
        return full_attr[idx]

    for i in range(batch_scene):
        K = repeat(
            torch.tensor(
                [
                    [0, 0, w / 2.0],
                    [0, 0, h / 2.0],
                    [0, 0, 1],
                ]
            ),
            "i j -> v i j",
            v=num_view,
        ).to(gaussian_means)
        K[:, 0, 0] = focal_length_x.reshape(batch_scene, num_view)[i]
        K[:, 1, 1] = focal_length_y.reshape(batch_scene, num_view)[i]

        i_means = index_i_gs_attr(gaussian_means, i)  # [N, 3]
        i_scales = index_i_gs_attr(gaussian_scales, i)
        i_quats = index_i_gs_attr(gaussian_quats, i)
        i_opacities = index_i_gs_attr(gaussian_opacities, i)  # [N,]
        i_colors = index_i_gs_attr(shs, i)  # [N, K, 3]
        i_viewmats = rearrange(view_matrix, "(b v) ... -> b v ...", v=num_view)[i]  # [v, 4, 4]
        i_backgrounds = rearrange(background_color, "(b v) ... -> b v ...", v=num_view)[i]  # [v, 3]

        render_colors, render_alphas, info = rasterization(
            means=i_means,
            quats=i_quats,  # [N, 4]
            scales=i_scales,  # [N, 3]
            opacities=i_opacities,
            colors=i_colors,
            viewmats=i_viewmats,  # [v, 4, 4]
            Ks=K,  # [v, 3, 3]
            backgrounds=i_backgrounds,
            render_mode=color_mode,
            width=w,
            height=h,
            packed=False,
            sh_degree=degree if use_sh else None,
        )
        depth = render_colors[..., -1].unbind(dim=0)

        image = rearrange(render_colors[..., :3], "v h w c -> v c h w").unbind(dim=0)
        radii = info["radii"].unbind(dim=0)
        try:
            info["means2d"].retain_grad()  # [1, N, 2]
        except Exception:
            pass
        all_images.extend(image)
        all_depths.extend(depth)
        all_radii.extend(radii)

    return torch.stack(all_images), torch.stack(all_depths)


def run_renderer_in_chunk_w_trj_mode(
    gaussians: Gaussians,
    extrinsics: torch.Tensor,  # world2cam, "batch view 4 4" | "batch view 3 4"
    intrinsics: torch.Tensor,  # unnormed intrinsics, "batch view 3 3"
    image_shape: tuple[int, int],
    chunk_size: Optional[int] = 8,
    trj_mode: Literal[
        "original",
        "smooth",
        "interpolate",
        "interpolate_smooth",
        "wander",
        "dolly_zoom",
        "extend",
        "wobble_inter",
    ] = "smooth",
    input_shape: Optional[tuple[int, int]] = None,
    enable_tqdm: Optional[bool] = False,
    **kwargs,
) -> tuple[
    torch.Tensor,  # color, "batch view 3 height width"
    torch.Tensor,  # depth, "batch view height width"
]:
    cam2world = affine_inverse(as_homogeneous(extrinsics))
    if input_shape is not None:
        in_h, in_w = input_shape
    else:
        in_h, in_w = image_shape
    intr_normed = intrinsics.clone().detach()
    intr_normed[..., 0, :] /= in_w
    intr_normed[..., 1, :] /= in_h
    if extrinsics.shape[1] <= 1:
        assert trj_mode in [
            "wander",
            "dolly_zoom",
        ], "Please set trj_mode to 'wander' or 'dolly_zoom' when n_views=1"

    def _smooth_trj_fn_batch(raw_c2ws, k_size=50):
        try:
            smooth_c2ws = torch.stack(
                [render_stabilization_path(c2w_i, k_size) for c2w_i in raw_c2ws],
                dim=0,
            )
        except Exception as e:
            print(f"[DEBUG] Path smoothing failed with error: {e}.")
            smooth_c2ws = raw_c2ws
        return smooth_c2ws

    # get rendered trj
    if trj_mode == "original":
        tgt_c2w = cam2world
        tgt_intr = intr_normed
    elif trj_mode == "smooth":
        tgt_c2w = _smooth_trj_fn_batch(cam2world)
        tgt_intr = intr_normed
    elif trj_mode in ["interpolate", "interpolate_smooth", "extend"]:
        inter_len = 8
        total_len = (cam2world.shape[1] - 1) * inter_len
        if total_len > 24 * 18:  # no more than 18s
            inter_len = max(1, 24 * 10 // (cam2world.shape[1] - 1))
        if total_len < 24 * 2:  # no less than 2s
            inter_len = max(1, 24 * 2 // (cam2world.shape[1] - 1))

        if inter_len > 2:
            t = torch.linspace(0, 1, inter_len, dtype=torch.float32, device=cam2world.device)
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
            tgt_c2w_b = []
            tgt_intr_b = []
            for b_idx in range(cam2world.shape[0]):
                tgt_c2w = []
                tgt_intr = []
                for cur_idx in range(cam2world.shape[1] - 1):
                    tgt_c2w.append(
                        interpolate_extrinsics(cam2world[b_idx, cur_idx], cam2world[b_idx, cur_idx + 1], t)[
                            (0 if cur_idx == 0 else 1) :
                        ]
                    )
                    tgt_intr.append(
                        interpolate_intrinsics(intr_normed[b_idx, cur_idx], intr_normed[b_idx, cur_idx + 1], t)[
                            (0 if cur_idx == 0 else 1) :
                        ]
                    )
                tgt_c2w_b.append(torch.cat(tgt_c2w))
                tgt_intr_b.append(torch.cat(tgt_intr))
            tgt_c2w = torch.stack(tgt_c2w_b)  # b v 4 4
            tgt_intr = torch.stack(tgt_intr_b)  # b v 3 3
        else:
            tgt_c2w = cam2world
            tgt_intr = intr_normed
        if trj_mode in ["interpolate_smooth", "extend"]:
            tgt_c2w = _smooth_trj_fn_batch(tgt_c2w)
        if trj_mode == "extend":
            # apply dolly_zoom and wander in the middle frame
            assert cam2world.shape[0] == 1, "extend only supports for batch_size=1 currently."
            mid_idx = tgt_c2w.shape[1] // 2
            c2w_wd, intr_wd = render_wander_path(
                tgt_c2w[0, mid_idx],
                tgt_intr[0, mid_idx],
                h=in_h,
                w=in_w,
                num_frames=max(36, min(60, mid_idx // 2)),
                max_disp=24.0,
            )
            c2w_dz, intr_dz = render_dolly_zoom_path(
                tgt_c2w[0, mid_idx],
                tgt_intr[0, mid_idx],
                h=in_h,
                w=in_w,
                num_frames=max(36, min(60, mid_idx // 2)),
            )
            tgt_c2w = torch.cat(
                [
                    tgt_c2w[:, :mid_idx],
                    c2w_wd.unsqueeze(0),
                    c2w_dz.unsqueeze(0),
                    tgt_c2w[:, mid_idx:],
                ],
                dim=1,
            )
            tgt_intr = torch.cat(
                [
                    tgt_intr[:, :mid_idx],
                    intr_wd.unsqueeze(0),
                    intr_dz.unsqueeze(0),
                    tgt_intr[:, mid_idx:],
                ],
                dim=1,
            )
    elif trj_mode in ["wander", "dolly_zoom"]:
        if trj_mode == "wander":
            render_fn = render_wander_path
            extra_kwargs = {"max_disp": 24.0}
        else:
            render_fn = render_dolly_zoom_path
            extra_kwargs = {"D_focus": 30.0, "max_disp": 2.0}
        tgt_c2w = []
        tgt_intr = []
        for b_idx in range(cam2world.shape[0]):
            c2w_i, intr_i = render_fn(cam2world[b_idx, 0], intr_normed[b_idx, 0], h=in_h, w=in_w, **extra_kwargs)
            tgt_c2w.append(c2w_i)
            tgt_intr.append(intr_i)
        tgt_c2w = torch.stack(tgt_c2w)
        tgt_intr = torch.stack(tgt_intr)
    elif trj_mode == "wobble_inter":
        tgt_c2w, tgt_intr = render_wobble_inter_path(
            cam2world=cam2world,
            intr_normed=intr_normed,
            inter_len=10,
            n_skip=3,
        )
    else:
        raise Exception(f"trj mode [{trj_mode}] is not implemented.")

    _, v = tgt_c2w.shape[:2]
    tgt_extr = affine_inverse(tgt_c2w)
    if chunk_size is None:
        chunk_size = v
    chunk_size = min(v, chunk_size)
    all_colors = []
    all_depths = []
    for chunk_idx in tqdm(
        range(math.ceil(v / chunk_size)),
        desc="Rendering novel views",
        disable=(not enable_tqdm),
        leave=False,
    ):
        s = int(chunk_idx * chunk_size)
        e = int((chunk_idx + 1) * chunk_size)
        cur_n_view = tgt_extr[:, s:e].shape[1]
        color, depth = render_3dgs(
            extrinsics=rearrange(tgt_extr[:, s:e], "b v ... -> (b v) ..."),  # w2c
            intrinsics=rearrange(tgt_intr[:, s:e], "b v ... -> (b v) ..."),  # normed
            image_shape=image_shape,
            gaussian=gaussians,
            num_view=cur_n_view,
            **kwargs,
        )
        all_colors.append(rearrange(color, "(b v) ... -> b v ...", v=cur_n_view))
        all_depths.append(rearrange(depth, "(b v) ... -> b v ...", v=cur_n_view))
    all_colors = torch.cat(all_colors, dim=1)
    all_depths = torch.cat(all_depths, dim=1)

    return all_colors, all_depths
