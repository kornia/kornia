from typing import Dict, Optional, Tuple

import torch

from kornia.color import rgb_to_grayscale
from kornia.core import Module, Tensor, concatenate, where, zeros_like
from kornia.feature import LocalFeatureMatcher, LoFTR
from kornia.geometry.homography import find_homography_dlt_iterated
from kornia.geometry.ransac import RANSAC
from kornia.geometry.transform import warp_perspective


class ImageStitcher(Module):
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

    def __init__(self, matcher: Module, estimator: str = "ransac", blending_method: str = "naive") -> None:
        super().__init__()
        self.matcher = matcher
        self.estimator = estimator
        self.blending_method = blending_method
        if estimator not in ["ransac", "vanilla"]:
            raise NotImplementedError(f"Unsupported estimator {estimator}. Use `ransac` or `vanilla` instead.")
        if estimator == "ransac":
            self.ransac = RANSAC("homography")

    def _estimate_homography(self, keypoints1: Tensor, keypoints2: Tensor) -> Tensor:
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

    def estimate_transform(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Compute the corresponding homography."""
        kp1, kp2, idx = kwargs["keypoints0"], kwargs["keypoints1"], kwargs["batch_indexes"]
        homos = [self._estimate_homography(kp1[idx == i], kp2[idx == i]) for i in range(len(idx.unique()))]

        if len(homos) == 0:
            raise RuntimeError("Compute homography failed. No matched keypoints found.")

        return concatenate(homos)

    def blend_image(self, src_img: Tensor, dst_img: Tensor, mask: Tensor) -> Tensor:
        """Blend two images together."""
        out: Tensor
        if self.blending_method == "naive":
            out = where(mask == 1, src_img, dst_img)
        else:
            raise NotImplementedError(f"Unsupported blending method {self.blending_method}. Use `naive`.")
        return out

    def preprocess(self, image_1: Tensor, image_2: Tensor) -> Dict[str, Tensor]:
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

    def postprocess(self, image: Tensor, mask: Tensor) -> Tensor:
        # NOTE: assumes no batch mode. This method keeps all valid regions after stitching.
        mask_ = mask.sum((0, 1))
        index = int(mask_.bool().any(0).long().argmin().item())
        if index == 0:  # If no redundant space
            return image
        return image[..., :index]

    def on_matcher(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.matcher(data)

    def stitch_pair(
        self,
        images_left: Tensor,
        images_right: Tensor,
        mask_left: Optional[Tensor] = None,
        mask_right: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Compute the transformed images
        input_dict = self.preprocess(images_left, images_right)
        out_shape = (images_left.shape[-2], images_left.shape[-1] + images_right.shape[-1])
        correspondences = self.on_matcher(input_dict)
        homo = self.estimate_transform(**correspondences)
        src_img = warp_perspective(images_right, homo, out_shape)
        dst_img = concatenate([images_left, zeros_like(images_right)], -1)

        # Compute the transformed masks
        if mask_left is None:
            mask_left = torch.ones_like(images_left)
        if mask_right is None:
            mask_right = torch.ones_like(images_right)
        # 'nearest' to ensure no floating points in the mask
        src_mask = warp_perspective(mask_right, homo, out_shape, mode="nearest")
        dst_mask = concatenate([mask_left, zeros_like(mask_right)], -1)
        return self.blend_image(src_img, dst_img, src_mask), (dst_mask + src_mask).bool().to(src_mask.dtype)

    def forward(self, *imgs: Tensor) -> Tensor:
        img_out = imgs[0]
        mask_left = torch.ones_like(img_out)
        for i in range(len(imgs) - 1):
            img_out, mask_left = self.stitch_pair(img_out, imgs[i + 1], mask_left)
        return self.postprocess(img_out, mask_left)
