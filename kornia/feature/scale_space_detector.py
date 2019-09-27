from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry import angle_to_rotation_matrix
from kornia.feature.responses import BlobHessian
from kornia.geometry import ConvSoftArgmax3d
from kornia.feature.orientation import PassLAF
from kornia.feature.laf import (
    denormalize_laf,
    normalize_laf)
from kornia.geometry.transform import ScalePyramid


def _scale_index_to_scale(max_coords: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Auxilary function for ScaleSpaceDetector. Converts scale level index from ConvSoftArgmax3d
    to the actual scale, using the sigmas from the ScalePyramid output
    Args:
        max_coords: (torch.Tensor): tensor [BxNx3].
        sigmas: (torch.Tensor): tensor [BxNxD], D >= 1

    Returns:
        torch.Tensor:  tensor [BxNx3].

    """
    # depth (scale) in coord_max is represented as (float) index, not the scale yet.
    # we will interpolate the scale using pytorch.grid_sample function
    # Because grid_sample is for 4d input only, we will create fake 2nd dimension
    # ToDo: replace with 3d input, when grid_sample will start to support it

    # Reshape for grid shape
    B, N, _ = max_coords.shape
    L: int = sigmas.size(1)
    scale_coords = max_coords[:, :, 0].view(-1, 1, 1, 1)

    # Normalize coordinate
    scale_coords_index = (2.0 * scale_coords / sigmas.size(1)) - 1.0

    # Dummy dimension
    dummy_x = torch.zeros_like(scale_coords_index)

    # Create grid
    scale_grid = torch.cat([scale_coords_index, dummy_x], dim=3)

    # Finally, interpolate the scale value
    scale_val = F.grid_sample(sigmas[0].view(1, 1, 1, -1).expand(scale_grid.size(0), 1, 1, L), scale_grid)

    # Replace the scale_x_y
    out = torch.cat([scale_val.view(B, N, 1), max_coords[:, :, 1:]], dim=2)
    return out


class ScaleSpaceDetector(nn.Module):
    def __init__(self,
                 num_features: int = 500,
                 mr_size: float = 5.192,
                 scale_pyr_module: nn.Module = ScalePyramid(3, 1.6, 10),
                 resp_module: nn.Module = BlobHessian(),
                 nms_module: nn.Module = ConvSoftArgmax3d((3, 3, 3),
                                                          (2, 2, 2),
                                                          (1, 1, 1),
                                                          normalized_coordinates=False,
                                                          output_value=True),
                 ori_module: nn.Module = PassLAF(),
                 aff_module: nn.Module = PassLAF()):
        super(ScaleSpaceDetector, self).__init__()
        self.mr_size = mr_size
        self.num_features = num_features
        self.scale_pyr = scale_pyr_module
        self.resp = resp_module
        self.nms = nms_module
        self.ori = ori_module
        self.aff = aff_module
        return

    def __repr__(self):
        return self.__class__.__name__ + '('\
            'num_features=' + str(self.num_features) + ', ' + \
            'mr_size=' + str(self.mr_size) + ', ' + \
            'scale_pyr=' + self.scale_pyr.__repr__() + ', ' + \
            'resp=' + self.resp.__repr__() + ', ' + \
            'nms=' + self.nms.__repr__() + ', ' + \
            'ori=' + self.ori.__repr__() + ', ' + \
            'aff=' + self.aff.__repr__() + ')'

    def detect(self, img: torch.Tensor, num_feats: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sp, sigmas, pix_dists = self.scale_pyr(img)
        all_responses = []
        all_lafs = []
        for oct_idx, octave in enumerate(sp):
            sigmas_oct = sigmas[oct_idx]
            pix_dists_oct = pix_dists[oct_idx]
            B, L, CH, H, W = octave.size()
            # Run response function
            oct_resp = self.resp(octave.view(B * L, CH, H, W), sigmas_oct.view(-1)).view(B, L, CH, H, W)

            # We want nms for scale responses, so reorder to (B, CH, L, H, W)
            oct_resp = oct_resp.permute(0, 2, 1, 3, 4)

            # Differentiable nms
            coord_max, response_max = self.nms(oct_resp)

            # Now, lets crop out some small responses
            responses_flatten = response_max.view(response_max.size(0), -1)  # [B * N, 3]
            max_coords_flatten = coord_max.view(response_max.size(0), 3, -1).permute(0, 2, 1)  # [B, N, 3]

            if responses_flatten.size(1) > num_feats:
                resp_flat_best, idxs = torch.topk(responses_flatten, k=num_feats, dim=1)
                max_coords_best = torch.gather(max_coords_flatten, 1, idxs.unsqueeze(-1).repeat(1, 1, 3))
            else:
                resp_flat_best = responses_flatten
                max_coords_best = max_coords_flatten
            B, N = resp_flat_best.size()

            # Converts scale level index from ConvSoftArgmax3d to the actual scale, using the sigmas
            max_coords_best = _scale_index_to_scale(max_coords_best, sigmas_oct)

            # Create local affine frames (LAFs)
            rotmat = angle_to_rotation_matrix(torch.zeros(B, N))
            current_lafs = torch.cat([self.mr_size * max_coords_best[:, :, 0].view(B, N, 1, 1) * rotmat,
                                      max_coords_best[:, :, 1:3].view(B, N, 2, 1)], dim=3)
            # Normalize LAFs
            current_lafs = normalize_laf(current_lafs, octave[:, 0])  # We don`t need # of scale levels, only shape

            all_responses.append(resp_flat_best)
            all_lafs.append(current_lafs)

        # Sort and keep best n
        responses: torch.Tensor = torch.cat(all_responses, dim=1)
        lafs: torch.Tensor = torch.cat(all_lafs, dim=1)
        responses, idxs = torch.topk(responses, k=num_feats, dim=1)
        lafs = lafs[:, idxs.view(-1), :, :]
        return responses, denormalize_laf(lafs, img)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        responses, lafs = self.detect(img, self.num_features)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        return lafs, responses
