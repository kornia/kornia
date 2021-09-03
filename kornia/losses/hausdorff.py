import torch
import torch.nn as nn


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion.

    This is an HD Loss that based on morphological erosion, which provided a differentiable
    approximation of Hausdorff distance as stated in https://arxiv.org/pdf/1904.10030.pdf.
    The code is modified on top of https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py.

    Args:
        alpha: controls the erosion rate in each iteration. Default: 2.0.
        erosions: the number of iterations of erosion. Default: 10.
    """

    def __init__(self, alpha: float = 2.0, erosions: int = 10) -> None:
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self) -> None:
        # Kernel from cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        cross = torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        self.kernel2D = cross * 0.2
        # TODO: implement 3D kernel
        # bound = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        # self.kernel3D = torch.stack([bound, cross, bound]) * (1 / 7)

    def perform_erosion(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        bound = (pred - target) ** 2

        if bound.ndim == 4:
            kernel = torch.as_tensor(self.kernel2D[:, None], device=pred.device, dtype=pred.dtype)
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = torch.zeros_like(bound)

        for k in range(self.erosions):

            # compute convolution with kernel
            padding = (kernel.size(2) - 1) // 2
            dilation = torch.conv2d(bound, weight=kernel, padding=padding, groups=1)
            # apply soft thresholding at 0.5 and normalize
            erosion = dilation - 0.5
            erosion[erosion < 0] = 0

            # image-wise differences for 2D images
            erosion_max = torch.nn.functional.adaptive_max_pool3d(erosion[:, None], (1, 1, 1)).squeeze()
            erosion_min = - torch.nn.functional.adaptive_max_pool3d(- erosion[:, None], (1, 1, 1)).squeeze()
            # No normalization needed if `max - min = 0`
            to_norm = (erosion_max - erosion_min) != 0
            erosion[to_norm] = (erosion[to_norm] - erosion_min[to_norm, None, None, None]) / (
                erosion_max[to_norm, None, None, None] - erosion_min[to_norm, None, None, None])

            # save erosion and add to loss
            eroted = eroted + erosion * (k + 1) ** self.alpha
            bound = erosion

        return eroted

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, c, x, y)
        target: (b, c, x, y)
        """
        assert pred.dim() == 4, f"Only 2D images supported. Got {pred.dim()}."
        assert pred.shape == target.shape, (
            "Prediction and target need to be of same shape."
            f"Got {pred.shape} and {target.shape}."
        )
        return torch.stack([
            self.perform_erosion(pred[:, i:i + 1], torch.where(target == i, 1, 0)).mean()
            for i in range(pred.size(1))
        ]).mean()
