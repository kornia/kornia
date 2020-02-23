from typing import Tuple, Union
from functools import reduce

import torch
import torch.nn as nn


class ZCAWhiten(nn.Module):
    r"""

    Computes ZCA whitening matrix transforms and mean vector and applies the transform
    to the data. The data tensor is flattened and the mean and covariance matrix are computed
    from the flattened data. The transformation is applied as a flattened matrix, and the output
    is resized to same size.


    args:

        eps (float) : a small number used for numerial stablility.
        biased (bool): Whether to use the biased estimate of the covariance matrix
        compute_inv (bool): Compute the inverse transform matrix

    Examples:

        >>> data = torch.tensor([[0,1],[1,0],[-1,0]], dtype = torch.float32)
        >>> zca = ZCAWhiten().fit(data)
        >>> data_whiten = zca(data)

    References:

        [1] 

    """

    def __init__(self, eps: float = 1e-8, biased: bool = False, compute_inv: bool =False) -> None:

        super(ZCAWhiten, self).__init__()

        self.eps: float = eps
        self.biased: bool = biased
        self.compute_inv: bool = compute_inv

    def fit(self, x: torch.Tensor): #type: ignore 
        r"""

        Fits ZCA whitening matrices to the data.

        args:

            x (torch.Tensor): Input data

        returns:
            ZCAWhiten: returns a ZCAWhiten object instance.
        """

        if self.compute_inv:
            T, T_inv, mean = zca_whiten_transforms(x, self.eps, self.biased, self.compute_inv)
            self.T_inv = T_inv
        else:
            T, mean = zca_whiten_transforms(x, self.eps, self.biased, self.compute_inv)
        self.mean: torch.Tensor = mean
        self.T: torch.Tensor = T

        return self

    def forward(self, x: torch.Tensor, inv_transform: bool = False) -> torch.Tensor:  # type: ignore
        r"""

        Applies the whitening transform to the data

        args:

            x (torch.Tensor): input data
            inv_transform (bool): If True, the ZCA transform is applied. Otherwise, the inverse transform is applied.

        returns:

            torch.Tensor : The transformed data

        """

        num_features: int = reduce(lambda a, b: a * b, x.size()[1::])

        x_flat: torch.Tensor = torch.reshape(x, (-1, num_features))

        if inv_transform:
            y: torch.Tensor = (x_flat).mm(self.T_inv.t()) + self.mean
        else:
            y = (x_flat - self.mean).mm(self.T.t())

        y = y.reshape(x.size())

        return y


def zca_whiten_transforms(inp: torch.Tensor, eps: float = 1e-8,
                          biased: bool = False, compute_inv: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                                                    Tuple[torch.Tensor, torch.Tensor,
                                                                                          torch.Tensor]]:
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
        Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A tuple containing the ZCA matrix and the mean vector, and if return_inv = True, the
        inverse whitening matrix is retured as well.



    """

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
    S_inv_root = S_inv_root

    T: torch.Tensor = U.mm(S_inv_root * U.t())

    if compute_inv:
        T_inv = U.t().mm(torch.sqrt(S) * U)
        return T, T_inv, mean
    else:
        return T, mean
