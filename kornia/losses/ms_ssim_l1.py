import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on:
# https://github.com/psyrocloud/MS-SSIM_L1_LOSS


class MS_SSIM_L1Loss(nn.Module):
    r"""Creates a criterion that computes MS-SSIM + L1 loss.

    According to [1], we compute the MS-SSIM + L1 loss as follows:

    .. math::
        \text{loss}(x, y) = \alpha \cdot \mathcal{L_{MS-SSIM}}(x,y)+(1 - \alpha) \cdot G_\alpha \cdot \mathcal{L_1}(x,y)

    Where:
        - :math:`\alpha` is the weight parameter.
        - :math:`x` and :math:`y` are the reconstructed and true reference images.
        - :math:`\mathcal{L_{MS-SSIM}}` is the MS-SSIM loss.
        - :math:`G_\alpha` is the sigma values for computing multi-scale SSIM.
        - :math:`\mathcal{L_1}` is the L1 loss.

    Reference:
        [1]: https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf#page11

    Args:
        sigmas: gaussian sigma values.
        data_range: the range of the images.
        K: k values.
        alpha : specifies the alpha value
        compensation: specifies the scaling coefficient.

    Returns:
        The computed loss.

    Shape:
        - Input: :math:`(N, C, H, W)`.
        - Output: :math:`(N,)` or scalar.

    Examples:
        >>> input1 = torch.rand(1, 3, 5, 5)
        >>> input2 = torch.rand(1, 3, 5, 5)
        >>> criterion = kornia.losses.MS_SSIM_L1Loss()
        >>> loss = criterion(input1, input2)
    """

    def __init__(self,
                 sigmas: list = [0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range: float = 1.0,
                 K: list = [0.01, 0.03],
                 alpha: float = 0.025,
                 compensation: float = 200.0) -> None:
        super().__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation

        # Set filter size
        filter_size = int(4 * sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(sigmas), 1, filter_size, filter_size))

        # Compute mask at different scales
        for idx, sigma in enumerate(sigmas):
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)

        self.g_masks = g_masks

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute MS-SSIM_L1 loss.

        Args:
            yhat: the predicted image with shape :math:`(B, C, H, W)`.
            y: the target image with a shape of :math:`(B, C, H, W)`.

        Returns:
            Estimated MS-SSIM_L1 loss.
        """
        if not isinstance(yhat, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(yhat)}")

        if not isinstance(y, torch.Tensor):
            raise TypeError(f"Output type is not a torch.Tensor. Got {type(y)}")

        if not len(yhat.shape) == len(y.shape):
            raise ValueError(f"Input shapes should be same. Got {type(yhat)} and {type(y)}.")

        if not yhat.device == y.device:
            raise ValueError(f"input and target must be in the same device. Got: {yhat.device} and {y.device}")

        self.g_masks = self.g_masks.to(yhat)
        b, c, h, w = yhat.shape
        mux = F.conv2d(yhat, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigmax2 = F.conv2d(yhat * yhat, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(yhat * y, self.g_masks, groups=3, padding=self.pad) - muxy
        lc = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        lM = lc[:, -1, :, :] * lc[:, -2, :, :] * lc[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        # Compute MS-SSIM loss
        loss_ms_ssim = 1 - lM * PIcs

        # Compute L1 loss
        loss_l1 = F.l1_loss(yhat, y, reduction='none')

        # Compute average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)

        # Compute MS-SSIM + L1 loss
        loss = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss = self.compensation * loss

        return loss.mean()
