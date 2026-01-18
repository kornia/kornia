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

# flake8: noqa E501
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

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from addict import Dict

from kornia.models.depth_anything_3.architecture.dpt import _make_fusion_block, _make_scratch
from kornia.models.depth_anything_3.architecture.utils.head_utils import (
    Permute,
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)


class DualDPT(nn.Module):
    """
    Dual-head DPT for dense prediction with an always-on auxiliary head.

    Architectural notes:
      - Sky/object branches are removed.
      - `intermediate_layer_idx` is fixed to (0, 1, 2, 3).
      - Auxiliary head has its **own** fusion blocks (no fusion_inplace / no sharing).
      - Auxiliary head is internally multi-level; **only the final level** is returned.
      - Returns a **dict** with keys from `head_names`, e.g.:
          { main_name, f"{main_name}_conf", aux_name, f"{aux_name}_conf" }
      - `feature_only` is fixed to False.
    """

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        aux_pyramid_levels: int = 4,
        aux_out1_conv_num: int = 5,
        head_names: Tuple[str, str] = ("depth", "ray"),
        **kwargs,
    ) -> None:
        super().__init__()

        # -------------------- configuration --------------------
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio

        self.aux_levels = aux_pyramid_levels
        self.aux_out1_conv_num = aux_out1_conv_num

        # names ONLY come from config (no hard-coded strings elsewhere)
        self.head_main, self.head_aux = head_names

        # Always expect 4 scales; enforce intermediate idx = (0, 1, 2, 3)
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        # -------------------- token pre-norm + per-stage projection --------------------
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # -------------------- spatial re-sizers (align to common scale before fusion) --------------------
        # design: stage strides (x4, x2, x1, /2) relative to patch grid to align to a common pivot scale
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        # -------------------- scratch: stage adapters + fusion (main & aux are separate) --------------------
        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain (independent)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # Primary head neck + head (independent)
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

        # Auxiliary fusion chain (completely separate; no sharing, i.e., "fusion_inplace=False")
        self.scratch.refinenet1_aux = _make_fusion_block(features)
        self.scratch.refinenet2_aux = _make_fusion_block(features)
        self.scratch.refinenet3_aux = _make_fusion_block(features)
        self.scratch.refinenet4_aux = _make_fusion_block(features, has_residual=False)

        # Aux pre-head per level (we will only *return final level*)
        self.scratch.output_conv1_aux = nn.ModuleList(
            [self._make_aux_out1_block(head_features_1) for _ in range(self.aux_levels)]
        )

        # Aux final projection per level
        use_ln = True
        ln_seq = [Permute((0, 2, 3, 1)), nn.LayerNorm(head_features_2), Permute((0, 3, 1, 2))] if use_ln else []
        self.scratch.output_conv2_aux = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                    *ln_seq,
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_features_2, 7, kernel_size=1, stride=1, padding=0),
                )
                for _ in range(self.aux_levels)
            ]
        )

    # -------------------------------------------------------------------------
    # Public forward (supports frame chunking for memory)
    # -------------------------------------------------------------------------

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            aggregated_tokens_list: List of 4 tensors [B, S, T, C] from transformer.
            images:                [B, S, 3, H, W], in [0, 1].
            patch_start_idx:       Patch-token start in the token sequence (to drop non-patch tokens).
            frames_chunk_size:     Optional chunking along S for memory.

        Returns:
            Dict[str, Tensor] with keys based on `head_names`, e.g.:
                self.head_main, f"{self.head_main}_conf",
                self.head_aux,  f"{self.head_aux}_conf"
            Shapes:
              main:    [B, S, out_dim, H/down_ratio, W/down_ratio]
              main_cf: [B, S, 1,       H/down_ratio, W/down_ratio]
              aux:     [B, S, 7,       H/down_ratio, W/down_ratio]
              aux_cf:  [B, S, 1,       H/down_ratio, W/down_ratio]
        """
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(feats, H, W, patch_start_idx)
            out_dict = {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return Dict(out_dict)
        out_dicts = []
        for s0 in range(0, B * S, chunk_size):
            s1 = min(s0 + chunk_size, B * S)
            out_dict = self._forward_impl(
                [feat[s0:s1] for feat in feats],
                H,
                W,
                patch_start_idx,
            )
            out_dicts.append(out_dict)
        out_dict = {k: torch.cat([out_dict[k] for out_dict in out_dicts], dim=0) for k in out_dicts[0].keys()}
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return Dict(out_dict)

    # -------------------------------------------------------------------------
    # Internal forward (single chunk)
    # -------------------------------------------------------------------------

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)  # align scales
            resized_feats.append(x)

        # 2) Fuse pyramid (main & aux are completely independent)
        fused_main, fused_aux_pyr = self._fuse(resized_feats)

        # 3) Upsample to target resolution and (optional) add pos-embed again
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused_main = custom_interpolate(fused_main, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)

        # Primary head: conv1 -> conv2 -> activate
        # fused_main = self.scratch.output_conv1(fused_main)
        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)

        # Auxiliary head (multi-level inside) -> only last level returned (after activation)
        last_aux = fused_aux_pyr[-1]
        if self.pos_embed:
            last_aux = self._add_pos_embed(last_aux, W, H)
        # neck (per-level pre-conv) then final projection (only for last level)
        # last_aux = self.scratch.output_conv1_aux[-1](last_aux)
        last_aux_logits = self.scratch.output_conv2_aux[-1](last_aux)
        fmap_last = last_aux_logits.permute(0, 2, 3, 1)
        aux_pred = self._apply_activation_single(fmap_last[..., :-1], "linear")
        aux_conf = self._apply_activation_single(fmap_last[..., -1], self.conf_activation)
        return {
            self.head_main: main_pred.squeeze(-1),
            f"{self.head_main}_conf": main_conf,
            self.head_aux: aux_pred,
            f"{self.head_aux}_conf": aux_conf,
        }

    # -------------------------------------------------------------------------
    # Subroutines
    # -------------------------------------------------------------------------

    def _fuse(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Feature pyramid fusion.
        Returns:
            fused_main: Tensor at finest scale (after refinenet1)
            aux_pyr:    List of aux tensors at each level (pre out_conv1_aux)
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # level 4 -> 3
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        aux_list: List[torch.Tensor] = []
        if self.aux_levels >= 4:
            aux_list.append(aux_out)

        # level 3 -> 2
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])
        if self.aux_levels >= 3:
            aux_list.append(aux_out)

        # level 2 -> 1
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])
        if self.aux_levels >= 2:
            aux_list.append(aux_out)

        # level 1 (final)
        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
        aux_list.append(aux_out)

        out = self.scratch.output_conv1(out)
        aux_list = [self.scratch.output_conv1_aux[i](aux) for i, aux in enumerate(aux_list)]

        return out, aux_list

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Simple UV positional embedding added to feature maps."""
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    def _make_aux_out1_block(self, in_ch: int) -> nn.Sequential:
        """Factory for the aux pre-head stack before the final 1x1 projection."""
        if self.aux_out1_conv_num == 5:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 3:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 1:
            return nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1))
        raise ValueError(f"aux_out1_conv_num {self.aux_out1_conv_num} not supported")

    def _apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        """
        Apply activation to single channel output, maintaining semantic consistency with value branch in multi-channel case.
        Supports: exp / relu / sigmoid / softplus / tanh / linear / expp1
        """
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        # Default linear
        return x


# # -----------------------------------------------------------------------------
# # Building blocks (tidy)
# # -----------------------------------------------------------------------------


# def _make_fusion_block(
#     features: int,
#     size: Tuple[int, int] = None,
#     has_residual: bool = True,
#     groups: int = 1,
#     inplace: bool = False,  # <- activation uses inplace=True by default; not related to "fusion_inplace"
# ) -> nn.Module:
#     return FeatureFusionBlock(
#         features=features,
#         activation=nn.ReLU(inplace=inplace),
#         deconv=False,
#         bn=False,
#         expand=False,
#         align_corners=True,
#         size=size,
#         has_residual=has_residual,
#         groups=groups,
#     )


# def _make_scratch(
#     in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False
# ) -> nn.Module:
#     scratch = nn.Module()
#     # optionally expand widths by stage
#     c1 = out_shape
#     c2 = out_shape * (2 if expand else 1)
#     c3 = out_shape * (4 if expand else 1)
#     c4 = out_shape * (8 if expand else 1)

#     scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
#     scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
#     scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
#     scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
#     return scratch


# class ResidualConvUnit(nn.Module):
#     """Lightweight residual conv block used within fusion."""

#     def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1) -> None:
#         super().__init__()
#         self.bn = bn
#         self.groups = groups
#         self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
#         self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
#         self.norm1 = None
#         self.norm2 = None
#         self.activation = activation
#         self.skip_add = nn.quantized.FloatFunctional()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
#         out = self.activation(x)
#         out = self.conv1(out)
#         if self.norm1 is not None:
#             out = self.norm1(out)

#         out = self.activation(out)
#         out = self.conv2(out)
#         if self.norm2 is not None:
#             out = self.norm2(out)

#         return self.skip_add.add(out, x)


# class FeatureFusionBlock(nn.Module):
#     """Top-down fusion block: (optional) residual merge + upsample + 1x1 shrink."""

#     def __init__(
#         self,
#         features: int,
#         activation: nn.Module,
#         deconv: bool = False,
#         bn: bool = False,
#         expand: bool = False,
#         align_corners: bool = True,
#         size: Tuple[int, int] = None,
#         has_residual: bool = True,
#         groups: int = 1,
#     ) -> None:
#         super().__init__()
#         self.align_corners = align_corners
#         self.size = size
#         self.has_residual = has_residual

#         self.resConfUnit1 = (
#             ResidualConvUnit(features, activation, bn, groups=groups) if has_residual else None
#         )
#         self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)

#         out_features = (features // 2) if expand else features
#         self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups)
#         self.skip_add = nn.quantized.FloatFunctional()

#     def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:  # type: ignore[override]
#         """
#         xs:
#           - xs[0]: top input
#           - xs[1]: (optional) lateral (to be added with residual)
#         """
#         y = xs[0]
#         if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
#             y = self.skip_add.add(y, self.resConfUnit1(xs[1]))

#         y = self.resConfUnit2(y)

#         # upsample
#         if (size is None) and (self.size is None):
#             up_kwargs = {"scale_factor": 2}
#         elif size is None:
#             up_kwargs = {"size": self.size}
#         else:
#             up_kwargs = {"size": size}

#         y = custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
#         y = self.out_conv(y)
#         return y
