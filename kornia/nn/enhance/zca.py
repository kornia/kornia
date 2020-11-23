from typing import Optional

import torch
import torch.nn as nn

import kornia


__all__ = [
    "ZCAWhitening",
]


class ZCAWhitening(nn.Module):
    r"""Compute the ZCA whitening matrix transform and the mean vector and applies the transform to the data.

    The data tensor is flattened, and the mean :math:`\mathbf{\mu}` and covariance matrix :math:`\mathbf{\Sigma}`
    are computed from the flattened data :math:`\mathbf{X} \in \mathbb{R}^{N \times D}`, where :math:`N` is the
    sample size and :math:`D` is flattened dimensionality (e.g. for a tensor with size 5x3x2x2 :math:`N = 5` and
    :math:`D = 12`). The ZCA whitening transform is given by:

    .. math::

        \mathbf{X}_{\text{zca}} = (\mathbf{X - \mu})(US^{-\frac{1}{2}}U^T)^T

    where :math:`U` are the eigenvectors of :math:`\Sigma` and :math:`S` contain the correpsonding
    eigenvalues of :math:`\Sigma`. After the transform is applied, the output is reshaped to same shape.

    Args:
        dim (int): Determines the dimension that represents the samples axis. Default = 0
        eps (float) : a small number used for numerial stablility. Default=1e-6
        unbiased (bool): Whether to use the biased estimate of the covariance matrix. Default=False
        compute_inv (bool): Compute the inverse transform matrix. Default=False
        detach_transforms (bool): Detaches gradient from the ZCA fitting. Default=True

    Shape:
        - x: :math:`(D_0,...,D_{\text{dim}},...,D_N)` is a batch of N-D tensors.
        - x_whiten: :math:`(D_0,...,D_{\text{dim}},...,D_N)` same shape as input.

    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype = torch.float32)
        >>> zca = kornia.color.ZCAWhitening().fit(x)
        >>> x_whiten = zca(x)
        >>> zca = kornia.color.ZCAWhitening()
        >>> x_whiten = zca(x, include_fit = True) # Includes the fitting step
        >>> x_whiten = zca(x) # Can run now without the fitting set
        >>> # Enable backprop through ZCA fitting process
        >>> zca = kornia.color.ZCAWhitening(detach_transforms = False)
        >>> x_whiten = zca(x, include_fit = True) # Includes the fitting step

    Note:
        This implementation uses :py:meth:`~torch.svd` which yields NaNs in the backwards step
        if the sigular values are not unique. See `here <https://pytorch.org/docs/stable/torch.html#torch.svd>`_ for
        more information.

    References:
        [1] `Stanford PCA & ZCA whitening tutorial <http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/>`_
    """

    def __init__(self, dim: int = 0, eps: float = 1e-6,
                 unbiased: bool = True, detach_transforms: bool = True,
                 compute_inv: bool = False) -> None:

        super(ZCAWhitening, self).__init__()

        self.dim = dim
        self.eps = eps
        self.unbiased = unbiased
        self.detach_transforms = detach_transforms
        self.compute_inv = compute_inv

        self.fitted = False

    def fit(self, x: torch.Tensor):
        r"""Fit ZCA whitening matrices to the data.

        Args:
            x (torch.Tensor): Input data.

        returns:
            ZCAWhiten: returns a fitted ZCAWhiten object instance.
        """

        T, mean, T_inv = kornia.enhance.zca_mean(x, self.dim, self.unbiased, self.eps, self.compute_inv)

        self.mean_vector: torch.Tensor = mean
        self.transform_matrix: torch.Tensor = T
        if T_inv is None:
            self.transform_inv: Optional[torch.Tensor] = torch.empty([0, ])
        else:
            self.transform_inv = T_inv

        if self.detach_transforms:
            self.mean_vector = self.mean_vector.detach()
            self.transform_matrix = self.transform_matrix.detach()
            self.transform_inv = self.transform_inv.detach()

        self.fitted = True

        return self

    def forward(self, x: torch.Tensor, include_fit: bool = False) -> torch.Tensor:
        r"""Apply the whitening transform to the data

        Args:
            x (torch.Tensor): Input data.
            include_fit (bool): Indicates whether to fit the data as part of the forward pass.

        Returns:
            torch.Tensor: The transformed data.
        """

        if include_fit:
            self.fit(x)

        if not self.fitted:
            raise RuntimeError("Needs to be fitted first before running. Please call fit or set include_fit to True.")

        x_whiten = kornia.enhance.linear_transform(x, self.transform_matrix, self.mean_vector, self.dim)

        return x_whiten

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply the inverse transform to the whitened data.

        Args:
            x (torch.Tensor): Whitened data.

        Returns:
            torch.Tensor: original data.
        """

        if not self.fitted:
            raise RuntimeError("Needs to be fitted first before running. Please call fit or set include_fit to True.")

        if not self.compute_inv:
            raise RuntimeError("Did not compute inverse ZCA. Please set compute_inv to True")

        mean_inv: torch.Tensor = -self.mean_vector.mm(self.transform_matrix)  # type: ignore

        y = kornia.enhance.linear_transform(x, self.transform_inv, mean_inv)  # type: ignore

        return y
