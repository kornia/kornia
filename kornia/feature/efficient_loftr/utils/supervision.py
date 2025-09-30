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

from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

from kornia.core import Tensor
from kornia.feature.loftr.utils.geometry import warp_kpts
from kornia.utils.grid import create_meshgrid


def static_vars(**kwargs: Dict[str, Any]) -> Callable[..., Any]:
    """Helper function."""

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func

    return decorate


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt: Tensor, mask: Tensor) -> Tensor:
    """For megadepth dataset, zero-padding exists in images."""
    # mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    mask = mask.view(mask.size(0), -1, 1).repeat(1, 1, 2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Coarse-level supervision.

    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    Note:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data["image0"].device
    N, _, H0, W0 = data["image0"].shape
    _, _, H1, W1 = data["image1"].shape
    scale = config["LOFTR"]["RESOLUTION"][0]
    scale0 = scale * data["scale0"][:, None] if "scale0" in data else scale
    scale1 = scale * data["scale1"][:, None] if "scale1" in data else scale
    h0, w0, h1, w1 = (x // scale for x in [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)  # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if "mask0" in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data["mask0"])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data["mask1"])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(grid_pt0_i, data["depth0"], data["depth1"], data["T_0to1"], data["K0"], data["K1"])
    _, w_pt1_i = warp_kpts(grid_pt1_i, data["depth1"], data["depth0"], data["T_1to0"], data["K1"], data["K0"])
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round()
    # calculate the overlap area between warped patch and grid patch as the loss weight.
    # (larger overlap area between warped patches and grid patch with higher weight)
    # (overlap area range from [0, 1] rather than [0.25, 1] as the penalty of warped kpts fall on
    # midpoint of two grid kpts)
    if config["LOFTR"]["LOSS"]["COARSE_OVERLAP_WEIGHT"]:
        w_pt0_c_error = (1.0 - 2 * torch.abs(w_pt0_c - w_pt0_c_round)).prod(-1)
    w_pt0_c_round = w_pt0_c_round[:, :, :].long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1

    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt: Tensor, w: Tensor, h: Tensor) -> Tensor:
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0 * w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({"conf_matrix_gt": conf_matrix_gt})

    # use overlap area as loss weight
    if config["LOFTR"]["LOSS"]["COARSE_OVERLAP_WEIGHT"]:
        conf_matrix_error_gt = w_pt0_c_error[b_ids, i_ids]  # weight range: [0.0, 1.0]
        data.update({"conf_matrix_error_gt": conf_matrix_error_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({"spv_b_ids": b_ids, "spv_i_ids": i_ids, "spv_j_ids": j_ids})

    # 6. save intermediate results (for fast fine-level computation)
    data.update({"spv_w_pt0_i": w_pt0_i, "spv_pt1_i": grid_pt1_i})


def compute_supervision_coarse(data: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Compute coarse supervision."""
    if len(set(data["dataset_name"])) != 1:
        raise AssertionError("Do not support mixed datasets training!")
    data_source = data["dataset_name"][0]
    if data_source.lower() in ["scannet", "megadepth"]:
        spvs_coarse(data, config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


@static_vars(counter={"value": 0})
@torch.no_grad()
def spvs_fine(data: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Fine-level supervision.

    Update:
    data (dict):{
        "expec_f_gt": [M, 2], used as subpixel-level gt
        "conf_matrix_f_gt": [M, WW, WW], M is the number of all coarse-level gt matches
        "conf_matrix_f_error_gt": [Mp], Mp is the number of all pixel-level gt matches
        "m_ids_f": [Mp]
        "i_ids_f": [Mp]
        "j_ids_f_di": [Mp]
        "j_ids_f_dj": [Mp]
        }
    """
    # 1. misc
    pt1_i = data["spv_pt1_i"]
    W = config["LOFTR"]["FINE_WINDOW_SIZE"]
    WW = W * W
    scale = config["LOFTR"]["RESOLUTION"][1]
    device = data["image0"].device
    N, _, H0, W0 = data["image0"].shape
    _, _, H1, W1 = data["image1"].shape
    hf0, wf0, _, _ = data["hw0_f"][0], data["hw0_f"][1], data["hw1_f"][0], data["hw1_f"][1]  # h, w of fine feature
    if not config["LOFTR"]["ALIGN_CORNER"]:
        raise ValueError("Only support training with align_corner=False for now.")

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data["b_ids"], data["i_ids"], data["j_ids"]
    scalei0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
    scalei1 = scale * data["scale1"][b_ids] if "scale1" in data else scale

    # 3. compute gt
    m = b_ids.shape[0]
    if m == 0:  # special case: there is no coarse gt
        conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device)

        data.update({"conf_matrix_f_gt": conf_matrix_f_gt})
        if config["LOFTR"]["LOSS"]["FINE_OVERLAP_WEIGHT"]:
            conf_matrix_f_error_gt = torch.zeros(1, device=device)
            data.update({"conf_matrix_f_error_gt": conf_matrix_f_error_gt})

        data.update({"expec_f": torch.zeros(1, 2, device=device)})
        data.update({"expec_f_gt": torch.zeros(1, 2, device=device)})
    else:
        grid_pt0_f = create_meshgrid(hf0, wf0, False, device) - W // 2 + 0.5  # [1, hf0, wf0, 2] # use fine coordinates
        # grid_pt0_f = rearrange(grid_pt0_f, 'n h w c -> n c h w')
        grid_pt0_f = grid_pt0_f.permute(0, 3, 1, 2)
        # 1. unfold(crop) all local windows
        if config["LOFTR"]["ALIGN_CORNER"] is False:  # even windows
            if W == 8:
                grid_pt0_f_unfold = F.unfold(grid_pt0_f, kernel_size=(W, W), stride=W, padding=0)
            else:
                raise ValueError("Value of W should be equal to 8.")
        # grid_pt0_f_unfold = rearrange(grid_pt0_f_unfold, 'n (c ww) l -> n l ww c', ww=W**2) # [1, hc0*wc0, W*W, 2]
        # grid_pt0_f_unfold = repeat(grid_pt0_f_unfold[0], 'l ww c -> N l ww c', N=N)
        n1, c_ww, l1 = grid_pt0_f_unfold.shape
        ww = W**2
        c = c_ww // ww
        grid_pt0_f_unfold = grid_pt0_f_unfold.reshape(n1, c, -1, l1).permute(0, 3, 2, 1)
        grid_pt0_f_unfold = grid_pt0_f_unfold[0].unsqueeze(0).repeat(N, 1, 1, 1)

        # 2. select only the predicted matches
        grid_pt0_f_unfold = grid_pt0_f_unfold[data["b_ids"], data["i_ids"]]  # [m, ww, 2]
        grid_pt0_f_unfold = scalei0[:, None, :] * grid_pt0_f_unfold  # [m, ww, 2]

        # 3. warp grids and get covisible & depth_consistent mask
        correct_0to1_f = torch.zeros(m, WW, device=device, dtype=torch.bool)
        w_pt0_i = torch.zeros(m, WW, 2, device=device, dtype=torch.float32)
        for b in range(N):
            mask = b_ids == b  # mask of each batch
            match = int(mask.sum())
            correct_0to1_f_mask, w_pt0_i_mask = warp_kpts(
                grid_pt0_f_unfold[mask].reshape(1, -1, 2),
                data["depth0"][[b], ...],
                data["depth1"][[b], ...],
                data["T_0to1"][[b], ...],
                data["K0"][[b], ...],
                data["K1"][[b], ...],
            )  # [k, WW], [k, WW, 2]
            correct_0to1_f[mask] = correct_0to1_f_mask.reshape(match, WW)
            w_pt0_i[mask] = w_pt0_i_mask.reshape(match, WW, 2)

        # 4. calculate the gt index of pixel-level refinement
        delta_w_pt0_i = w_pt0_i - pt1_i[b_ids, j_ids][:, None, :]  # [m, WW, 2]
        del b_ids, i_ids, j_ids
        delta_w_pt0_f = delta_w_pt0_i / scalei1[:, None, :] + W // 2 - 0.5
        delta_w_pt0_f_round = delta_w_pt0_f[:, :, :].round()
        if config["LOFTR"]["LOSS"]["FINE_OVERLAP_WEIGHT"]:
            # calculate the overlap area between warped patch and grid patch as the loss weight.
            w_pt0_f_error = (1.0 - 2 * torch.abs(delta_w_pt0_f - delta_w_pt0_f_round)).prod(-1)  # [0, 1]
        delta_w_pt0_f_round = delta_w_pt0_f_round.long()

        nearest_index1 = delta_w_pt0_f_round[..., 0] + delta_w_pt0_f_round[..., 1] * W  # [m, WW]

        # corner case: out of fine windows
        def out_bound_mask(pt: Tensor, w: Tensor, h: Tensor) -> Tensor:
            return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

        ob_mask = out_bound_mask(delta_w_pt0_f_round, W, W)
        nearest_index1[ob_mask] = 0
        correct_0to1_f[ob_mask] = 0

        m_ids, i_ids = torch.where(correct_0to1_f != 0)
        j_ids = nearest_index1[m_ids, i_ids]  # i_ids, j_ids range from [0, WW-1]
        # further get the (i, j) index in fine windows of image1 (right image);
        # j_ids_di, j_ids_dj range from [0, W-1]
        j_ids_di, j_ids_dj = (
            j_ids // W,
            j_ids % W,
        )
        m_ids, i_ids, j_ids_di, j_ids_dj = (
            m_ids.to(torch.long),
            i_ids.to(torch.long),
            j_ids_di.to(torch.long),
            j_ids_dj.to(torch.long),
        )

        # expec_f_gt will be used as the gt of subpixel-level refinement
        expec_f_gt = delta_w_pt0_f - delta_w_pt0_f_round

        if m_ids.numel() == 0:  # special case: there is no pixel-level gt
            # this won't affect fine-level loss calculation
            data.update({"expec_f": torch.zeros(1, 2, device=device)})
            data.update({"expec_f_gt": torch.zeros(1, 2, device=device)})
        else:
            expec_f_gt = expec_f_gt[m_ids, i_ids]
            data.update({"expec_f_gt": expec_f_gt})
            data.update({"m_ids_f": m_ids, "i_ids_f": i_ids, "j_ids_f_di": j_ids_di, "j_ids_f_dj": j_ids_dj})

        # 5. construct a pixel-level gt conf_matrix
        conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device, dtype=torch.bool)
        conf_matrix_f_gt[m_ids, i_ids, j_ids] = 1
        data.update({"conf_matrix_f_gt": conf_matrix_f_gt})
        if config["LOFTR"]["LOSS"]["FINE_OVERLAP_WEIGHT"]:
            # calculate the overlap area between warped pixel and grid pixel as the loss weight.
            w_pt0_f_error = w_pt0_f_error[m_ids, i_ids]
            data.update({"conf_matrix_f_error_gt": w_pt0_f_error})


def compute_supervision_fine(data: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Compute fine supervision."""
    data_source = data["dataset_name"][0]
    if data_source.lower() in ["scannet", "megadepth"]:
        spvs_fine(data, config)
    else:
        raise NotImplementedError
