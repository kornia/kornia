from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from kornia.geometry.transform.imgwarp import warp_perspective
from kornia.geometry.linalg import transform_points
from kornia.geometry.ransac import RANSAC
from kornia.geometry.subpix import ConvQuadInterp3d

from kornia.testing import check_is_tensor
from kornia.feature import (
    LocalFeatureMatcher,
    ScaleSpaceDetector,
    CornerGFTT,
    HardNet,
    LAFAffNetShapeEstimator,
    LAFOrienter,
    DescriptorMatcher,
    LoFTR,
)


__all__ = [
    "HomographyTracker",
]


class HomographyTracker(nn.Module):
    r"""Module, which performs local-feature-based tracking of the target planar object in the
    sequence of the frames..
    Args:
      sdsd

    Example:
        >>> from kornia.geometry import ImageRegistrator
        >>> img_src = torch.rand(1, 1, 32, 32)
        >>> img_dst = torch.rand(1, 1, 32, 32)
        >>> registrator = ImageRegistrator('similarity')
        >>> homo = registrator.register(img_src, img_dst)
    """
    def __init__(self,
                 initial_matcher = LocalFeatureMatcher(ScaleSpaceDetector(2000,
                                                                          resp_module=CornerGFTT(),
                                                                          nms_module=ConvQuadInterp3d(10, 2e-4),
                                                                          mr_size=6.0,
                                                                          aff_module=LAFAffNetShapeEstimator(32),
                                                                          ori_module=LAFOrienter(patch_size=19)),

                                                        HardNet(True),
                                                        DescriptorMatcher('smnn', 0.95)),
                 fast_matcher = LoFTR('outdoor'),
                 ransac =  RANSAC('homography',
                                   inl_th = 5.0,
                                   batch_size = 4096,
                                   max_iter = 10,
                                   max_lo_iters = 10),
                                   minimum_inliers_num: int = 30) -> None:
      super().__init__()
      self.initial_matcher = initial_matcher
      self.fast_matcher = fast_matcher
      self.ransac = ransac
      self.minimum_inliers_num = minimum_inliers_num
      self.reset_tracking()

    def set_target(self, target):
      self.target = target
      self.target_initial_representation = {}
      self.target_fast_representation = {}
      if hasattr(self.initial_matcher, 'extract_features'):
        self.target_initial_representation = self.initial_matcher.extract_features(target)

      if hasattr(self.fast_matcher, 'extract_features'):
        self.target_fast_representation = self.fast_matcher.extract_features(target)
      return


    def reset_tracking(self):
      self.previous_homography = None

    def no_match(self):
      return None, False

    def match_initial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      input_dict ={"image0": self.target,
                   "image1": x}
      for k, v in self.target_initial_representation.items():
        input_dict[f'{k}0'] = v
      match_dict = self.initial_matcher(input_dict)
      keypoints0 = match_dict['keypoints0'][match_dict['batch_indexes'] == 0]
      keypoints1 = match_dict['keypoints1'][match_dict['batch_indexes'] == 0]
      if len(keypoints0) < self.minimum_inliers_num:
        return self.no_match()
      H, inliers = self.ransac(keypoints0, keypoints1)
      if inliers.sum().item() < self.minimum_inliers_num:
        return self.no_match()
      self.previous_homography = H.clone()
      return H, True

    def track_next_frame(self, x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        Hwarp = self.previous_homography.clone()[None]
        # make a bit of border for safety
        Hwarp[:, 0:2, 0:2] = Hwarp[:, 0:2, 0:2] / 0.8
        Hwarp[:, 0:2, 2] += -10.0
        Hinv = torch.inverse(Hwarp)
        frame_warped = warp_perspective(x, Hinv, self.target.shape[2:])
        input_dict ={"image0": self.target,
                     "image1": frame_warped}
        for k, v in self.target_fast_representation.items():
          input_dict[f'{k}0'] = v

        match_dict = self.fast_matcher(input_dict)
        keypoints0 = match_dict['keypoints0'][match_dict['batch_indexes'] == 0]
        keypoints1 = match_dict['keypoints1'][match_dict['batch_indexes'] == 0]
        keypoints1 = transform_points(Hwarp, keypoints1)

        if len(keypoints0) < self.minimum_inliers_num:
          self.reset_tracking()
          return self.no_match()
        H, inliers = self.ransac(keypoints0, keypoints1)
        if inliers.sum().item() < self.minimum_inliers_num:
          self.reset_tracking()
          return self.no_match()
        self.previous_homography = H.clone()
        return H, True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      if self.previous_homography is not None:
        return self.track_next_frame(x)
      return self.match_initial(x)
