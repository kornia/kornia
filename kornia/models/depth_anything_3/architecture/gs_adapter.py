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

from typing import Optional

import torch
from einops import einsum, rearrange, repeat
from torch import nn

from kornia.models.depth_anything_3.specs import Gaussians
from kornia.models.depth_anything_3.utils.geometry import affine_inverse, get_world_rays, sample_image_grid
from kornia.models.depth_anything_3.utils.pose_align import batch_align_poses_umeyama
from kornia.models.depth_anything_3.utils.sh_helpers import rotate_sh

from .utils.transform import cam_quat_xyzw_to_world_quat_wxyz


class GaussianAdapter(nn.Module):
    def __init__(
        self,
        sh_degree: int = 0,
        pred_color: bool = False,
        pred_offset_depth: bool = False,
        pred_offset_xy: bool = True,
        gaussian_scale_min: float = 1e-5,
        gaussian_scale_max: float = 30.0,
        **kwargs,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.pred_color = pred_color
        self.pred_offset_depth = pred_offset_depth
        self.pred_offset_xy = pred_offset_xy
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        if not pred_color:
            self.register_buffer(
                "sh_mask",
                torch.ones((self.d_sh,), dtype=torch.float32),
                persistent=False,
            )
            for degree in range(1, sh_degree + 1):
                self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: torch.Tensor,  # "*#batch 4 4"
        intrinsics: torch.Tensor,  # "*#batch 3 3"
        depths: torch.Tensor,  # "*#batch"
        opacities: torch.Tensor,  # "*#batch" | "*#batch _"
        raw_gaussians: torch.Tensor,  # "*#batch _"
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        gt_extrinsics: Optional[torch.Tensor] = None,  # "*#batch 4 4"
        **kwargs,
    ) -> Gaussians:
        device = extrinsics.device
        dtype = raw_gaussians.dtype
        H, W = image_shape
        b, v = raw_gaussians.shape[:2]

        # get cam2worlds and intr_normed to adapt to 3DGS codebase
        cam2worlds = affine_inverse(extrinsics)
        intr_normed = intrinsics.clone().detach()
        intr_normed[..., 0, :] /= W
        intr_normed[..., 1, :] /= H

        # 1. compute 3DGS means
        # 1.1) offset the predicted depth if needed
        if self.pred_offset_depth:
            gs_depths = depths + raw_gaussians[..., -1]
            raw_gaussians = raw_gaussians[..., :-1]
        else:
            gs_depths = depths
        # 1.2) align predicted poses with GT if needed
        if gt_extrinsics is not None and not torch.equal(extrinsics, gt_extrinsics):
            try:
                _, _, pose_scales = batch_align_poses_umeyama(
                    gt_extrinsics.detach().float(),
                    extrinsics.detach().float(),
                )
            except Exception:
                pose_scales = torch.ones_like(extrinsics[:, 0, 0, 0])
            pose_scales = torch.clamp(pose_scales, min=1 / 3.0, max=3.0)
            cam2worlds[:, :, :3, 3] = cam2worlds[:, :, :3, 3] * rearrange(pose_scales, "b -> b () ()")  # [b, i, j]
            gs_depths = gs_depths * rearrange(pose_scales, "b -> b () () ()")  # [b, v, h, w]
        # 1.3) casting xy in image space
        xy_ray, _ = sample_image_grid((H, W), device)
        xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # b v h w xy
        # offset xy if needed
        if self.pred_offset_xy:
            pixel_size = 1 / torch.tensor((W, H), dtype=xy_ray.dtype, device=device)
            offset_xy = raw_gaussians[..., :2]
            xy_ray = xy_ray + offset_xy * pixel_size
            raw_gaussians = raw_gaussians[..., 2:]  # skip the offset_xy
        # 1.4) unproject depth + xy to world ray
        origins, directions = get_world_rays(
            xy_ray,
            repeat(cam2worlds, "b v i j -> b v h w i j", h=H, w=W),
            repeat(intr_normed, "b v i j -> b v h w i j", h=H, w=W),
        )
        gs_means_world = origins + directions * gs_depths[..., None]
        gs_means_world = rearrange(gs_means_world, "b v h w d -> b (v h w) d")

        # 2. compute other GS attributes
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        # 2.1) 3DGS scales
        # make the scale invarient to resolution
        scale_min = self.gaussian_scale_min
        scale_max = self.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        pixel_size = 1 / torch.tensor((W, H), dtype=dtype, device=device)
        multiplier = self.get_scale_multiplier(intr_normed, pixel_size)
        gs_scales = scales * gs_depths[..., None] * multiplier[..., None, None, None]
        gs_scales = rearrange(gs_scales, "b v h w d -> b (v h w) d")

        # 2.2) 3DGS quaternion (world space)
        # due to historical issue, assume quaternion in order xyzw, not wxyz
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        # rotate them to world space
        cam_quat_xyzw = rearrange(rotations, "b v h w c -> b (v h w) c")
        c2w_mat = repeat(
            cam2worlds,
            "b v i j -> b (v h w) i j",
            h=H,
            w=W,
        )
        world_quat_wxyz = cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w_mat)
        gs_rotations_world = world_quat_wxyz  # b (v h w) c

        # 2.3) 3DGS color / SH coefficient (world space)
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        if not self.pred_color:
            sh = sh * self.sh_mask

        if self.pred_color or self.sh_degree == 0:
            # predict pre-computed color or predict only DC band, no need to transform
            gs_sh_world = sh
        else:
            gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
        gs_sh_world = rearrange(gs_sh_world, "b v h w xyz d_sh -> b (v h w) xyz d_sh")

        # 2.4) 3DGS opacity
        gs_opacities = rearrange(opacities, "b v h w ... -> b (v h w) ...")

        return Gaussians(
            means=gs_means_world,
            harmonics=gs_sh_world,
            opacities=gs_opacities,
            scales=gs_scales,
            rotations=gs_rotations_world,
        )

    def get_scale_multiplier(
        self,
        intrinsics: torch.Tensor,  # "*#batch 3 3"
        pixel_size: torch.Tensor,  # "*#batch 2"
        multiplier: float = 0.1,
    ) -> torch.Tensor:  # " *batch"
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].float().inverse().to(intrinsics),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return 1 if self.pred_color else (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        # provided as reference to the gs_dpt output dim
        raw_gs_dim = 0
        if self.pred_offset_xy:
            raw_gs_dim += 2
        raw_gs_dim += 3  # scales
        raw_gs_dim += 4  # quaternion
        raw_gs_dim += 3 * self.d_sh  # color
        if self.pred_offset_depth:
            raw_gs_dim += 1

        return raw_gs_dim
