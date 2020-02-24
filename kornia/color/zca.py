from typing import Tuple
from functools import reduce

import torch
import torch.nn as nn


class ZCAWhiten(nn.Module):
    r"""

    Computes the ZCA whitening matrix transform and the mean vector and applies the transform
    to the data. The data tensor is flattened and the mean and covariance matrix are computed
    from the flattened data. The transformation is applied as a flattened matrix, and the output
    is reshaped to same shape.


    args:

        eps (float) : a small number used for numerial stablility. Default=1e-7
        biased (bool): Whether to use the biased estimate of the covariance matrix. Default=False
        compute_inv (bool): Compute the inverse transform matrix. Default=False
        detach_transforms (bool): Detaches gradient from the ZCA fitting. Default=True

    shape:
        - x: :math:`(N, *)` is a batch of N-D tensors.
        - x_whiten: :math:`(N, *)`


    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0]], dtype = torch.float32)
        >>> zca = ZCAWhiten().fit(x)
        >>> x_whiten = zca(data)

    Note:

        This implementation uses :class:`~torch.svd` which yields NaNs in the backwards step
        if the sigular values are not unique.

    References:

        [1] `Stanford PCA & ZCA whitening tutorial <http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/>`_

    """

    def __init__(self, eps: float = 1e-7, biased: bool = False, compute_inv: bool = False,
                 detach_transforms: bool = True) -> None:

        super(ZCAWhiten, self).__init__()

        self.eps: float = eps
        self.biased: bool = biased
        self.compute_inv: bool = compute_inv
        self.detach_transforms: bool = detach_transforms
        self.fitted = False

    def fit(self, x: torch.Tensor):  # type: ignore
        r"""

        Fits ZCA whitening matrices to the data.

        args:

            x (torch.Tensor): Input data

        returns:
            ZCAWhiten: returns a fitted ZCAWhiten object instance.
        """

        if self.compute_inv:
            T, T_inv, mean = zca_whiten_transforms(x, self.eps, self.biased, self.compute_inv)
            self.T_inv = T_inv

            if self.detach_transforms:
                self.T_inv = self.T_inv.detach()

        else:
            T, mean = zca_whiten_transforms(x, self.eps, self.biased, self.compute_inv)

        self.mean: torch.Tensor = mean
        self.T: torch.Tensor = T

        if self.detach_transforms:
            self.mean = self.mean.detach()
            self.T = self.T.detach()

        self.fitted = True

        return self

    def forward(self, x: torch.Tensor, include_fit: bool = False) -> torch.Tensor:  # type: ignore
        r"""

        Applies the whitening transform to the data

        args:

            x (torch.Tensor): Input data
            include_fit (bool): Indicates whether to fit the data as part of the forward pass

        returns:

            torch.Tensor : The transformed data

        """

        if include_fit:
            self.fit(x)

        if not self.fitted:
            raise RuntimeError("Needs to be fitted first before running. Please call fit or set include_fit to True.")

        num_features: int = reduce(lambda a, b: a * b, x.size()[1::])

        x_flat: torch.Tensor = torch.reshape(x, (-1, num_features))

        y = (x_flat - self.mean).mm(self.T)

        y = y.reshape(x.size())

        return y

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        Applies the inverse transform to the whitened data.

        args:
            x (torch.Tensor): Whitened data
            include_fit (bool): Indicates whether to fit the data as part of the forward pass

        returns:
            torch.Tensor: original data



        """

        if not self.fitted:
            raise RuntimeError("Needs to be fitted first before running. Please call fit or set include_fit to True.")
        if not self.compute_inv:
            raise RuntimeError("Tried to use inverse_transform without computing the inverse transform matrix.")

        num_features: int = reduce(lambda a, b: a * b, x.size()[1::])

        x_flat: torch.Tensor = torch.reshape(x, (-1, num_features))

        y: torch.Tensor = (x_flat).mm(self.T_inv) + self.mean

        y = y.reshape(x.size())

        return y


def zca_whiten_transforms(inp: torch.Tensor, eps: float = 1e-7,
                          biased: bool = False, compute_inv: bool = False) -> Tuple[torch.Tensor, ...]:
    r"""

    Computes ZCA whitening matrix and mean vector. The output could be used in
    :class:`~torchvision.transforms.LinearTransformation`.

    See :class:`~kornia.color.ZCAWhiten` for details.


    args:
        inp (torch.Tensor) : input data tensor
        eps (float) : a small number used for numerial stablility.
        biased (bool): Whether to use the biased estimate of the covariance matrix
        compute_inv (bool): Whether to return the inverse transform matrix


    returns:
        Tuple[torch.Tensor, ...]:
        A tuple containing the ZCA matrix and the mean vector, and if compute_inv = True, the
        inverse whitening matrix is retured as well.



    """

    if not torch.is_tensor(inp):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(inp)))

    N: int = inp.size()[0]
    num_features: int = reduce(lambda a, b: a * b, inp.size()[1::])

    mean: torch.Tensor = torch.mean(inp, dim=0, keepdim=True)
    mean = mean.view((1, num_features))

    inp_center_flat: torch.Tensor = inp.view((N, num_features)) - mean

    cov = inp_center_flat.t().mm(inp_center_flat)

    if biased:
        cov = cov / float(N)
    else:
        cov = cov / float(N - 1)

    U, S, _ = torch.svd(cov)

    S = S.view(-1, 1)
    S_inv_root: torch.Tensor = torch.rsqrt(S + eps)
    T: torch.Tensor = (U).mm(S_inv_root * U.t())

    if compute_inv:
        T_inv: torch.Tensor = (U).mm(torch.sqrt(S) * U.t())
        return T, T_inv, mean
    else:
        return T, mean
