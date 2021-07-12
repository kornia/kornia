from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from kornia.utils.helpers import _torch_svd_cast

__all__ = ["zca_mean", "zca_whiten", "linear_transform", "ZCAWhitening"]


class ZCAWhitening(nn.Module):
    r"""Computes the ZCA whitening matrix transform and the mean vector and applies the transform to the data.

    The data tensor is flattened, and the mean :math:`\mathbf{\mu}`
    and covariance matrix :math:`\mathbf{\Sigma}` are computed from
    the flattened data :math:`\mathbf{X} \in \mathbb{R}^{N \times D}`, where
    :math:`N` is the sample size and :math:`D` is flattened dimensionality
    (e.g. for a tensor with size 5x3x2x2 :math:`N = 5` and :math:`D = 12`). The ZCA whitening
    transform is given by:

    .. math::

        \mathbf{X}_{\text{zca}} = (\mathbf{X - \mu})(US^{-\frac{1}{2}}U^T)^T

    where :math:`U` are the eigenvectors of :math:`\Sigma` and :math:`S` contain the corresponding
    eigenvalues of :math:`\Sigma`. After the transform is applied, the output is reshaped to same shape.

    args:

        dim: Determines the dimension that represents the samples axis.
        eps: a small number used for numerical stability.
        unbiased: Whether to use the biased estimate of the covariance matrix.
        compute_inv: Compute the inverse transform matrix.
        detach_transforms: Detaches gradient from the ZCA fitting.

    shape:
        - x: :math:`(D_0,...,D_{\text{dim}},...,D_N)` is a batch of N-D tensors.
        - x_whiten: :math:`(D_0,...,D_{\text{dim}},...,D_N)` same shape as input.

    .. note::
       See a working example `here <https://colab.sandbox.google.com/github/kornia/tutorials/
       blob/master/source/zca_whitening.ipynb>`__.

    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype = torch.float32)
        >>> zca = ZCAWhitening().fit(x)
        >>> x_whiten = zca(x)
        >>> zca = ZCAWhitening()
        >>> x_whiten = zca(x, include_fit = True) # Includes the fitting step
        >>> x_whiten = zca(x) # Can run now without the fitting set
        >>> # Enable backprop through ZCA fitting process
        >>> zca = ZCAWhitening(detach_transforms = False)
        >>> x_whiten = zca(x, include_fit = True) # Includes the fitting step

    Note:

        This implementation uses :py:meth:`~torch.svd` which yields NaNs in the backwards step
        if the singular values are not unique. See `here <https://pytorch.org/docs/stable/torch.html#torch.svd>`_ for
        more information.

    References:

        [1] `Stanford PCA & ZCA whitening tutorial <http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/>`_
    """

    def __init__(
        self,
        dim: int = 0,
        eps: float = 1e-6,
        unbiased: bool = True,
        detach_transforms: bool = True,
        compute_inv: bool = False,
    ) -> None:

        super(ZCAWhitening, self).__init__()

        self.dim = dim
        self.eps = eps
        self.unbiased = unbiased
        self.detach_transforms = detach_transforms
        self.compute_inv = compute_inv

        self.fitted = False

    def fit(self, x: torch.Tensor):
        r"""Fits ZCA whitening matrices to the data.

        Args:

            x: Input data.

        returns:
            Returns a fitted ZCAWhiten object instance.
        """

        T, mean, T_inv = zca_mean(x, self.dim, self.unbiased, self.eps, self.compute_inv)

        self.mean_vector: torch.Tensor = mean
        self.transform_matrix: torch.Tensor = T
        if T_inv is None:
            self.transform_inv: Optional[torch.Tensor] = torch.empty([0])
        else:
            self.transform_inv = T_inv

        if self.detach_transforms:
            self.mean_vector = self.mean_vector.detach()
            self.transform_matrix = self.transform_matrix.detach()
            self.transform_inv = self.transform_inv.detach()

        self.fitted = True

        return self

    def forward(self, x: torch.Tensor, include_fit: bool = False) -> torch.Tensor:
        r"""Applies the whitening transform to the data.

        Args:
            x: Input data.
            include_fit: Indicates whether to fit the data as part of the forward pass.

        Returns:
            The transformed data.

        """

        if include_fit:
            self.fit(x)

        if not self.fitted:
            raise RuntimeError("Needs to be fitted first before running. Please call fit or set include_fit to True.")

        x_whiten = linear_transform(x, self.transform_matrix, self.mean_vector, self.dim)

        return x_whiten

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        r"""Applies the inverse transform to the whitened data.

        Args:
            x: Whitened data.

        Returns:
            Original data.
        """

        if not self.fitted:
            raise RuntimeError("Needs to be fitted first before running. Please call fit or set include_fit to True.")

        if not self.compute_inv:
            raise RuntimeError("Did not compute inverse ZCA. Please set compute_inv to True")

        mean_inv: torch.Tensor = -self.mean_vector.mm(self.transform_matrix)  # type: ignore

        y = linear_transform(x, self.transform_inv, mean_inv)  # type: ignore

        return y


def zca_mean(
    inp: torch.Tensor, dim: int = 0, unbiased: bool = True, eps: float = 1e-6, return_inverse: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""Computes the ZCA whitening matrix and mean vector.

    The output can be used with :py:meth:`~kornia.color.linear_transform`.
    See :class:`~kornia.color.ZCAWhitening` for details.

    Args:
        inp: input data tensor.
        dim: Specifies the dimension that serves as the samples dimension.
        unbiased: Whether to use the unbiased estimate of the covariance matrix.
        eps: a small number used for numerical stability.
        return_inverse: Whether to return the inverse ZCA transform.

    Shapes:
        - inp: :math:`(D_0,...,D_{\text{dim}},...,D_N)` is a batch of N-D tensors.
        - transform_matrix: :math:`(\Pi_{d=0,d\neq \text{dim}}^N D_d, \Pi_{d=0,d\neq \text{dim}}^N D_d)`
        - mean_vector: :math:`(1, \Pi_{d=0,d\neq \text{dim}}^N D_d)`
        - inv_transform: same shape as the transform matrix

    Returns:
        A tuple containing the ZCA matrix and the mean vector. If return_inverse is set to True,
        then it returns the inverse ZCA matrix, otherwise it returns None.

    .. note::
       See a working example `here <https://colab.sandbox.google.com/github/kornia/tutorials/
       blob/master/source/zca_whitening.ipynb>`__.

    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype = torch.float32)
        >>> transform_matrix, mean_vector,_ = zca_mean(x) # Returns transformation matrix and data mean
        >>> x = torch.rand(3,20,2,2)
        >>> transform_matrix, mean_vector, inv_transform = zca_mean(x, dim = 1, return_inverse = True)
        >>> # transform_matrix.size() equals (12,12) and the mean vector.size equal (1,12)

    """

    if not isinstance(inp, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(inp)))

    if not isinstance(eps, float):
        raise TypeError(f"eps type is not a float. Got{type(eps)}")

    if not isinstance(unbiased, bool):
        raise TypeError(f"unbiased type is not bool. Got{type(unbiased)}")

    if not isinstance(dim, int):
        raise TypeError("Argument 'dim' must be of type int. Got {}".format(type(dim)))

    if not isinstance(return_inverse, bool):
        raise TypeError("Argument return_inverse must be of type bool {}".format(type(return_inverse)))

    inp_size = inp.size()

    if dim >= len(inp_size) or dim < -len(inp_size):
        raise IndexError(
            "Dimension out of range (expected to be in range of [{},{}], but got {}".format(
                -len(inp_size), len(inp_size) - 1, dim
            )
        )

    if dim < 0:
        dim = len(inp_size) + dim

    feat_dims = torch.cat([torch.arange(0, dim), torch.arange(dim + 1, len(inp_size))])

    new_order: List[int] = torch.cat([torch.tensor([dim]), feat_dims]).tolist()

    inp_permute = inp.permute(new_order)

    N = inp_size[dim]
    feature_sizes = torch.tensor(inp_size[0:dim] + inp_size[dim + 1 : :])
    num_features: int = int(torch.prod(feature_sizes).item())

    mean: torch.Tensor = torch.mean(inp_permute, dim=0, keepdim=True)

    mean = mean.reshape((1, num_features))

    inp_center_flat: torch.Tensor = inp_permute.reshape((N, num_features)) - mean

    cov = inp_center_flat.t().mm(inp_center_flat)

    if unbiased:
        cov = cov / float(N - 1)
    else:
        cov = cov / float(N)

    U, S, _ = _torch_svd_cast(cov)

    S = S.reshape(-1, 1)
    S_inv_root: torch.Tensor = torch.rsqrt(S + eps)
    T: torch.Tensor = (U).mm(S_inv_root * U.t())

    T_inv: Optional[torch.Tensor] = None
    if return_inverse:
        T_inv = (U).mm(torch.sqrt(S + eps) * U.t())

    return T, mean, T_inv


def zca_whiten(inp: torch.Tensor, dim: int = 0, unbiased: bool = True, eps: float = 1e-6) -> torch.Tensor:
    r"""Applies ZCA whitening transform.

    See :class:`~kornia.color.ZCAWhitening` for details.

    Args:
        inp: input data tensor.
        dim: Specifies the dimension that serves as the samples dimension.
        unbiased: Whether to use the unbiased estimate of the covariance matrix.
        eps: a small number used for numerical stability.

    Returns:
        Whiten Input data.

    .. note::
       See a working example `here <https://colab.sandbox.google.com/github/kornia/tutorials/
       blob/master/source/zca_whitening.ipynb>`__.

    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0]], dtype = torch.float32)
        >>> zca_whiten(x)
        tensor([[ 0.0000,  1.1547],
                [ 1.0000, -0.5773],
                [-1.0000, -0.5773]])
    """

    if not isinstance(inp, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(inp)))

    if not isinstance(eps, float):
        raise TypeError(f"eps type is not a float. Got{type(eps)}")

    if not isinstance(unbiased, bool):
        raise TypeError(f"unbiased type is not bool. Got{type(unbiased)}")

    if not isinstance(dim, int):
        raise TypeError("Argument 'dim' must be of type int. Got {}".format(type(dim)))

    transform, mean, _ = zca_mean(inp, dim, unbiased, eps, False)

    inp_whiten = linear_transform(inp, transform, mean, dim)

    return inp_whiten


def linear_transform(
    inp: torch.Tensor, transform_matrix: torch.Tensor, mean_vector: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    r"""

    Given a transformation matrix and a mean vector, this function will flatten
    the input tensor along the given dimension and subtract the mean vector
    from it. Then the dot product with the transformation matrix will be computed
    and then the resulting tensor is reshaped to the original input shape.

    .. math::

        \mathbf{X}_{T} = (\mathbf{X - \mu})(T)

    Args:
        inp: Input data :math:`X`.
        transform_matrix: Transform matrix :math:`T`.
        mean_vector: mean vector :math:`\mu`.
        dim: Batch dimension.

    Shapes:
        - inp: :math:`(D_0,...,D_{\text{dim}},...,D_N)` is a batch of N-D tensors.
        - transform_matrix: :math:`(\Pi_{d=0,d\neq \text{dim}}^N D_d, \Pi_{d=0,d\neq \text{dim}}^N D_d)`
        - mean_vector: :math:`(1, \Pi_{d=0,d\neq \text{dim}}^N D_d)`

    Returns:
        Transformed data.

    Example:
        >>> # Example where dim = 3
        >>> inp = torch.ones((10,3,4,5))
        >>> transform_mat = torch.ones((10*3*4,10*3*4))
        >>> mean = 2*torch.ones((1,10*3*4))
        >>> out = linear_transform(inp, transform_mat, mean, 3)
        >>> print(out.shape, out.unique())  # Should a be (10,3,4,5) tensor of -120s
        torch.Size([10, 3, 4, 5]) tensor([-120.])

        >>> # Example where dim = 0
        >>> inp = torch.ones((10,2))
        >>> transform_mat = torch.ones((2,2))
        >>> mean = torch.zeros((1,2))
        >>> out = linear_transform(inp, transform_mat, mean)
        >>> print(out.shape, out.unique()) # Should a be (10,2) tensor of 2s
        torch.Size([10, 2]) tensor([2.])
    """

    inp_size = inp.size()

    if dim >= len(inp_size) or dim < -len(inp_size):
        raise IndexError(
            "Dimension out of range (expected to be in range of [{},{}], but got {}".format(
                -len(inp_size), len(inp_size) - 1, dim
            )
        )

    if dim < 0:
        dim = len(inp_size) + dim

    feat_dims = torch.cat([torch.arange(0, dim), torch.arange(dim + 1, len(inp_size))])

    perm = torch.cat([torch.tensor([dim]), feat_dims])
    perm_inv = torch.argsort(perm)

    new_order: List[int] = perm.tolist()
    inv_order: List[int] = perm_inv.tolist()

    N = inp_size[dim]
    feature_sizes = torch.tensor(inp_size[0:dim] + inp_size[dim + 1 : :])
    num_features: int = int(torch.prod(feature_sizes).item())

    inp_permute = inp.permute(new_order)
    inp_flat = inp_permute.reshape((-1, num_features))

    inp_center = inp_flat - mean_vector
    inp_transformed = inp_center.mm(transform_matrix)

    inp_transformed = inp_transformed.reshape(inp_permute.size())

    inp_transformed = inp_transformed.permute(inv_order)

    return inp_transformed
