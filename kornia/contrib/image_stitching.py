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

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from kornia.color import rgb_to_grayscale
from kornia.feature import LocalFeatureMatcher, LoFTR
from kornia.geometry.homography import find_homography_dlt_iterated
from kornia.geometry.ransac import RANSAC
from kornia.geometry.transform import warp_perspective


class ImageStitcher(nn.Module):
    """Stitch two images with overlapping fields of view.

    Args:
        matcher: image feature matching module.
        estimator: method to compute homography, either "vanilla" or "ransac".
            "ransac" is slower with a better accuracy.
        blending_method: method to blend two images together.
            Only "naive" is currently supported.

    Note:
        Current implementation requires strict image ordering from left to right.

    .. code-block:: python

        IS = ImageStitcher(KF.LoFTR(pretrained='outdoor'), estimator='ransac').cuda()
        # Compute the stitched result with less GPU memory cost.
        with torch.inference_mode():
            out = IS(img_left, img_right)
        # Show the result
        plt.imshow(K.tensor_to_image(out))

    """

    def __init__(self, matcher: nn.Module, estimator: str = "ransac", blending_method: str = "naive") -> None:
        super().__init__()
        self.matcher = matcher
        self.estimator = estimator
        self.blending_method = blending_method
        if estimator not in ["ransac", "vanilla"]:
            raise NotImplementedError(f"Unsupported estimator {estimator}. Use `ransac` or `vanilla` instead.")
        if estimator == "ransac":
            self.ransac = RANSAC("homography")

    def _estimate_homography(self, keypoints1: torch.Tensor, keypoints2: torch.Tensor) -> torch.Tensor:
        """Estimate homography by the matched keypoints.

        Args:
            keypoints1: matched keypoint set from an image, shaped as :math:`(N, 2)`.
            keypoints2: matched keypoint set from the other image, shaped as :math:`(N, 2)`.

        """
        if self.estimator == "vanilla":
            homo = find_homography_dlt_iterated(
                keypoints2[None], keypoints1[None], torch.ones_like(keypoints1[None, :, 0])
            )
        elif self.estimator == "ransac":
            homo, _ = self.ransac(keypoints2, keypoints1)
            homo = homo[None]
        else:
            raise NotImplementedError(f"Unsupported estimator {self.estimator}. Use `ransac` or `vanilla` instead.")
        return homo

    def estimate_transform(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Compute the corresponding homography."""
        kp1, kp2, idx = kwargs["keypoints0"], kwargs["keypoints1"], kwargs["batch_indexes"]
        homos = [self._estimate_homography(kp1[idx == i], kp2[idx == i]) for i in range(len(idx.unique()))]

        if len(homos) == 0:
            raise RuntimeError("Compute homography failed. No matched keypoints found.")

        return torch.cat(homos)

    def blend_image(self, src_img: torch.Tensor, dst_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Blend two images together."""
        out: torch.Tensor
        if self.blending_method == "naive":
            out = torch.where(mask == 1, src_img, dst_img)
        else:
            raise NotImplementedError(f"Unsupported blending method {self.blending_method}. Use `naive`.")
        return out

    def preprocess(self, image_1: torch.Tensor, image_2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Preprocess input to the required format."""
        # TODO: probably perform histogram matching here.
        if isinstance(self.matcher, (LoFTR, LocalFeatureMatcher)):
            input_dict = {  # LofTR works on grayscale images only
                "image0": rgb_to_grayscale(image_1),
                "image1": rgb_to_grayscale(image_2),
            }
        else:
            raise NotImplementedError(f"The preprocessor for {self.matcher} has not been implemented.")
        return input_dict

    def postprocess(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Crop redundant empty columns from a stitched panorama.

        Args:
            image: Stitched image tensor, typically shaped :math:`(B, C, H, W_{panorama})`.
            mask: Validity mask aligned with ``image`` where non-zero values mark
                pixels covered by at least one input view.

        Returns:
            The stitched image cropped to remove trailing invalid columns.
            If no redundant area is found, the original ``image`` is returned.
        """
        # NOTE: assumes no batch mode. This method keeps all valid regions after stitching.
        mask_ = mask.sum((0, 1))
        index = int(mask_.bool().any(0).long().argmin().item())
        if index == 0:  # If no redundant space
            return image
        return image[..., :index]

    def on_matcher(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the configured feature matcher on preprocessed inputs.

        Args:
            data: Matcher input dictionary, usually containing ``image0`` and ``image1``.

        Returns:
            A correspondence dictionary produced by ``self.matcher``.
        """
        return self.matcher(data)

    def stitch_pair(
        self,
        images_left: torch.Tensor,
        images_right: torch.Tensor,
        mask_left: Optional[torch.Tensor] = None,
        mask_right: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stitch two images into a shared panorama frame.

        Args:
            images_left: Reference image tensor, shaped :math:`(B, C, H, W_l)`.
            images_right: Image tensor to be warped onto the reference frame,
                shaped :math:`(B, C, H, W_r)`.
            mask_left: Optional validity mask for ``images_left``.
            mask_right: Optional validity mask for ``images_right``.

        Returns:
            A tuple ``(stitched_image, stitched_mask)`` where:
            - ``stitched_image`` is the blended panorama tensor.
            - ``stitched_mask`` marks valid pixels in the panorama.
        """
        # Compute the transformed images
        input_dict = self.preprocess(images_left, images_right)
        out_shape = (images_left.shape[-2], images_left.shape[-1] + images_right.shape[-1])
        correspondences = self.on_matcher(input_dict)
        homo = self.estimate_transform(**correspondences)
        src_img = warp_perspective(images_right, homo, out_shape)
        dst_img = torch.cat([images_left, torch.zeros_like(images_right)], -1)

        # Compute the transformed masks
        if mask_left is None:
            mask_left = torch.ones_like(images_left)
        if mask_right is None:
            mask_right = torch.ones_like(images_right)
        # 'nearest' to ensure no floating points in the mask
        src_mask = warp_perspective(mask_right, homo, out_shape, mode="nearest")
        dst_mask = torch.cat([mask_left, torch.zeros_like(mask_right)], -1)
        return self.blend_image(src_img, dst_img, src_mask), (dst_mask + src_mask).bool().to(src_mask.dtype)

    def forward(self, *imgs: torch.Tensor) -> torch.Tensor:
        """Iteratively stitch a sequence of overlapping images.

        Args:
            *imgs: Ordered image tensors from left to right. Each tensor is expected
                to share a common height and overlap with the next image.

        Returns:
            A single panorama tensor obtained by stitching all provided images.
        """
        img_out = imgs[0]
        mask_left = torch.ones_like(img_out)
        for i in range(len(imgs) - 1):
            img_out, mask_left = self.stitch_pair(img_out, imgs[i + 1], mask_left)
        return self.postprocess(img_out, mask_left)
