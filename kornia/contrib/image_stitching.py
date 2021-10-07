import torch
import torch.nn as nn

import kornia as K

__all__ = ["ImageStitching"]


class ImageStitching(nn.Module):
    """Stitch two images with overlapping fields of view.

    Args:
        matcher: image feature matching module.
        blending_method: method to blend two images together.
            Only "naive" is currently supported.

    Note:
        Current implementation requires strict image ordering from left to right.

    Example:
        >>> IS = ImageStitching(K.feature.LoFTR(pretrained='outdoor'))
    """
    def __init__(self, matcher: nn.Module, blending_method: str = "naive"):
        super().__init__()
        self.matcher = matcher
        self.blending_method = blending_method

    def _find_homography(self, keypoints1: torch.Tensor, keypoints2: torch.Tensor) -> torch.Tensor:
        homo = K.find_homography_dlt_iterated(
            keypoints1[None],
            keypoints2[None],
            torch.ones_like(keypoints1[None, :, 0])
        )
        return homo

    def compute_homo_from_results(self, **kwargs) -> torch.Tensor:
        """Compute the corresponding homography."""
        homos = []
        kp1, kp2, idx = kwargs['keypoints0'], kwargs['keypoints1'], kwargs['batch_indexes']
        for i in range(len(idx.unique())):
            homos.append(self._find_homography(kp1[idx == i], kp2[idx == i]))
        return torch.cat(homos)

    def compute_mask(self, image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
        """Compute the image blending mask."""
        return torch.cat([torch.zeros_like(image_1), torch.ones_like(image_2)], dim=-1)

    def blend_image(self, src_img: torch.Tensor, dst_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """blend two images together."""
        if self.blending_method == "naive":
            out = torch.where(mask == 1, src_img, dst_img)
            return out
        raise NotImplementedError

    def preprocessing(self, image_1: torch.Tensor, image_2: torch.Tensor):
        """Preprocess input to the required format."""
        # TODO: probably perform histogram matching here.
        if isinstance(self.matcher, K.feature.LoFTR):
            input_dict = {  # LofTR works on grayscale images only
                "image0": K.color.rgb_to_grayscale(image_1),
                "image1": K.color.rgb_to_grayscale(image_2)
            }
            return input_dict
        raise NotImplementedError

    def forward(self, image_left: torch.Tensor, image_right: torch.Tensor) -> torch.Tensor:
        input = self.preprocessing(image_right, image_left)
        mask = self.compute_mask(image_right, image_left)
        with torch.no_grad():
            correspondences = self.matcher(input)
        homo = self.compute_homo_from_results(**correspondences)
        src_img = K.warp_perspective(image_right, homo, (mask.shape[-2], mask.shape[-1]))
        dst_img = torch.cat([image_left, torch.zeros_like(image_right)], dim=-1)
        return self.blend_image(src_img, dst_img, mask)
