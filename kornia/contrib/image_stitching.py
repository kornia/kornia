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
    """Stitch images with overlapping fields of view.

    Convention:
        - The images are RGB, :math:`(B, 3, H, W)`, all of one height, ordered left to right. ``matcher`` is a
          :class:`~kornia.feature.LoFTR` or a :class:`~kornia.feature.LocalFeatureMatcher`, which gets grayscale
          copies; any other matcher raises ``NotImplementedError``.
        - The panorama is in the first image's pixel frame and has its height: content that maps left of its first
          column, or above or below its rows, is lost. Each next image is warped onto the panorama so far with a
          homography from their matches; every sample needs 4 matches, or a ``RuntimeError`` names the sample.
        - ``"naive"`` blending keeps the warped image wherever it covers a pixel completely, so in an overlap the right
          image replaces the left one; a pixel it covers only in part keeps the panorama's value.
        - The result drops the empty trailing columns, those after the last column that any sample covers; an
          uncovered column before it, the first one included, is kept. A batch shares one crop, so it is cropped to its
          widest sample and a narrower sample keeps zero columns on its right.
        - ``estimator="ransac"`` uses ``RANSAC("homography")`` (``inl_th=2.0``); ``"vanilla"`` fits every match with
          iterated DLT.

    Args:
        matcher: image feature matching module, :class:`~kornia.feature.LoFTR` or
            :class:`~kornia.feature.LocalFeatureMatcher`.
        estimator: method to compute homography, either "vanilla" or "ransac".
            "ransac" is slower with a better accuracy.
        blending_method: method to blend two images together.
            Only "naive" is currently supported.

    .. code-block:: python

        import torch
        import kornia.feature as KF
        from kornia.contrib import ImageStitcher

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stitcher = ImageStitcher(KF.LoFTR(pretrained="outdoor"), estimator="ransac").to(device)
        # img_left and img_right: (1, 3, H, W) RGB images of one height on `device`
        with torch.inference_mode():
            panorama = stitcher(img_left, img_right)

    See :doc:`/applications/image_stitching` for a complete example.

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

    def estimate_transform(
        self, *args: torch.Tensor, batch_size: Optional[int] = None, **kwargs: torch.Tensor
    ) -> torch.Tensor:
        """Compute the corresponding homography of every sample.

        Args:
            *args: ignored.
            batch_size: number of samples. Defaults to one more than the largest batch index of the matches.
            **kwargs: the matcher output, with ``keypoints0``, ``keypoints1`` and ``batch_indexes``.

        Raises:
            RuntimeError: if a sample has fewer than 4 matches (the error names the sample), if a batch index is outside
                the batch (the error names the index), or if there are no matches and ``batch_size`` is not given.
        """
        kp1, kp2, idx = kwargs["keypoints0"], kwargs["keypoints1"], kwargs["batch_indexes"]
        if batch_size is None:
            if idx.numel() == 0:
                raise RuntimeError("Compute homography failed. No matched keypoints found.")
            n = int(idx.max().item()) + 1
        else:
            n = batch_size
        outside = idx[(idx < 0) | (idx >= n)]
        if outside.numel() > 0:
            bad = int(outside[0].item())
            if bad < 0:
                raise RuntimeError(f"Compute homography failed: batch index {bad} is negative.")
            samples = "1 sample" if n == 1 else f"{n} samples"
            raise RuntimeError(f"Compute homography failed: batch index {bad} is outside the batch of {samples}.")
        selections = [idx == i for i in range(n)]
        for i, sel in enumerate(selections):
            n_i = int(sel.sum().item())
            if n_i < 4:
                raise RuntimeError(f"Compute homography failed for sample {i}: {n_i} matches, a homography needs 4.")
        homos = [self._estimate_homography(kp1[sel], kp2[sel]) for sel in selections]
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
            image: Stitched image tensor, typically shaped
                :math:`(B, C, H, W_{panorama})`, where ``W_{panorama}`` is the
                panorama width after concatenation/warping.
            mask: Validity mask aligned with ``image`` where non-zero values mark
                pixels covered by at least one input view.

        Returns:
            The stitched image without its empty trailing columns, those after the last column that has a valid pixel
            in any sample of the mask. Uncovered columns before it are kept, and a mask without any valid pixel returns
            the original ``image``.
        """
        # One crop for the whole batch: a column is valid if any sample has a valid pixel in it.
        mask_ = mask.sum((0, 1))
        index = int(mask_.bool().any(0).flip(0).long().argmax().item())
        if index == 0:  # If no redundant space
            return image
        return image[..., :-index]

    def on_matcher(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the configured feature matcher on preprocessed inputs.

        Args:
            data: Matcher input dictionary, usually containing ``image0`` and
                ``image1`` tensors.

        Returns:
            A correspondence dictionary produced by ``self.matcher``.
            Typical entries are matched keypoints and batch indices used later
            for homography estimation.
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
            mask_left: Optional validity mask for ``images_left`` with the same
                shape as ``images_left``.
            mask_right: Optional validity mask for ``images_right`` with the same
                shape as ``images_right``.

        Returns:
            A tuple ``(stitched_image, stitched_mask)``:
            - ``stitched_image`` is the blended panorama tensor with width
              ``W_l + W_r`` before final cropping.
            - ``stitched_mask`` is a mask tensor indicating which panorama
              pixels are valid after warping and blending.
        """
        # Compute the transformed images
        input_dict = self.preprocess(images_left, images_right)
        out_shape = (images_left.shape[-2], images_left.shape[-1] + images_right.shape[-1])
        correspondences = self.on_matcher(input_dict)
        homo = self.estimate_transform(batch_size=images_left.shape[0], **correspondences)
        dst_img = torch.cat([images_left, torch.zeros_like(images_right)], -1)

        # Compute the transformed masks
        if mask_left is None:
            mask_left = torch.ones_like(images_left)
        if mask_right is None:
            mask_right = torch.ones_like(images_right)
        # The image is resampled bilinearly, so a panorama pixel whose footprint reaches past the right image's border
        # mixes in the zero padding. Warp the validity mask together with the image, in one call and so on the same
        # sampling grid, and keep only the pixels covered completely.
        # The warp runs in at least float32 (float64 stays float64), whatever the dtype of the image and of the
        # homography: a half sampling grid is coarser than a pixel on a wide panorama. Autocast is disabled around it,
        # which would otherwise run the homography matmuls in half precision, and the float32 coverage tolerance below
        # would then reject completely covered border pixels.
        n_ch = images_right.shape[1]
        work = torch.promote_types(images_right.dtype, torch.float32)
        with torch.autocast(device_type=images_right.device.type, enabled=False):
            warped = warp_perspective(
                torch.cat([images_right.to(work), mask_right.to(work)], 1), homo.to(work), out_shape
            )
        src_img = warped[:, :n_ch].to(images_right.dtype)
        # Round-off in the warp's sampling positions grows with the coordinates and puts a completely covered border
        # pixel past the border, so its coverage reads below 1. Measured: up to 1.5 * eps * max(H_out, W_out) for
        # integer shifts (0.99854 on the last row and column of a 4096 x 4096 image shifted by 2048), and up to
        # 2.86 * eps * max(H_out, W_out) for general homographies. The tolerance scales the same way, with a margin,
        # and stays far below a genuinely partial coverage: in float32 it reaches 0.3 only past 600 000 px.
        tol = max(1e-4, 4.0 * torch.finfo(work).eps * max(out_shape))
        src_mask = (warped[:, n_ch:] >= 1.0 - tol).to(images_right.dtype)
        dst_mask = torch.cat([mask_left, torch.zeros_like(mask_right)], -1)
        return self.blend_image(src_img, dst_img, src_mask), (dst_mask + src_mask).bool().to(src_mask.dtype)

    def forward(self, *imgs: torch.Tensor) -> torch.Tensor:
        """Iteratively stitch a sequence of overlapping images.

        Args:
            *imgs: Ordered image tensors from left to right. Each tensor is expected
                to share a common height and overlap with the next image.
                Each tensor is RGB, :math:`(B, 3, H, W)`.

        Returns:
            A single panorama tensor obtained by stitching all provided images
            in order and cropping it as :meth:`postprocess` does.
        """
        img_out = imgs[0]
        mask_left = torch.ones_like(img_out)
        for i in range(len(imgs) - 1):
            img_out, mask_left = self.stitch_pair(img_out, imgs[i + 1], mask_left)
        return self.postprocess(img_out, mask_left)
