import torch
import torch.nn as nn

import kornia as K

__all__ = ["ImageStitcher"]


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

    def __init__(self, matcher: nn.Module, estimator: str = 'ransac', blending_method: str = "naive") -> None:
        super().__init__()
        self.matcher = matcher
        self.estimator = estimator
        self.blending_method = blending_method
        if estimator not in ['ransac', 'vanilla']:
            raise NotImplementedError
        if estimator == "ransac":
            self.ransac = K.geometry.RANSAC('homography')

    def _estimate_homography(self, keypoints1: torch.Tensor, keypoints2: torch.Tensor) -> torch.Tensor:
        """Estimate homography by the matched keypoints.

        Args:
            keypoints1: matched keypoint set from an image, shaped as :math:`(N, 2)`.
            keypoints2: matched keypoint set from the other image, shaped as :math:`(N, 2)`.
        """
        if self.estimator == "vanilla":
            homo = K.find_homography_dlt_iterated(
                keypoints2[None],
                keypoints1[None],
                torch.ones_like(keypoints1[None, :, 0])
            )
        elif self.estimator == "ransac":
            homo, _ = self.ransac(keypoints2, keypoints1)
            homo = homo[None]
        else:
            raise NotImplementedError
        return homo

    def estimate_transform(self, **kwargs) -> torch.Tensor:
        """Compute the corresponding homography."""
        homos = []
        kp1, kp2, idx = kwargs['keypoints0'], kwargs['keypoints1'], kwargs['batch_indexes']
        for i in range(len(idx.unique())):
            homos.append(self._estimate_homography(kp1[idx == i], kp2[idx == i]))
        if len(homos) == 0:
            raise RuntimeError("Compute homography failed. No matched keypoints found.")
        return torch.cat(homos)

    def compute_mask(self, image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
        """Compute the image blending mask."""
        return torch.cat([torch.zeros_like(image_1), torch.ones_like(image_2)], dim=-1)

    def blend_image(self, src_img: torch.Tensor, dst_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Blend two images together."""
        if self.blending_method == "naive":
            out = torch.where(mask == 1, src_img, dst_img)
        else:
            raise NotImplementedError
        return out

    def preprocess(self, image_1: torch.Tensor, image_2: torch.Tensor):
        """Preprocess input to the required format."""
        # TODO: probably perform histogram matching here.
        if isinstance(self.matcher, K.feature.LoFTR):
            input_dict = {  # LofTR works on grayscale images only
                "image0": K.color.rgb_to_grayscale(image_1),
                "image1": K.color.rgb_to_grayscale(image_2)
            }
        else:
            raise NotImplementedError
        return input_dict

    def forward(self, images_left: torch.Tensor, images_right: torch.Tensor) -> torch.Tensor:
        # TODO: accept a list of images for composing paranoma
        input = self.preprocess(images_left, images_right)
        mask = self.compute_mask(images_left, images_right)
        correspondences = self.matcher(input)
        homo = self.estimate_transform(**correspondences)
        src_img = K.geometry.warp_perspective(images_right, homo, (mask.shape[-2], mask.shape[-1]))
        dst_img = torch.cat([images_left, torch.zeros_like(images_right)], dim=-1)
        return self.blend_image(src_img, dst_img, mask)
