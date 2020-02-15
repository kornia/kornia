from typing import Tuple, Union
from functools import reduce

import torch 
import torch.nn as nn


def zca_whiten(inp: torch.Tensor, eps: float = 1e-8,
              biased: bool = False, return_inv: bool = False
              ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    r"""

    Computes ZCA whitening matrix and mean vector. 


    args:
        inp (torch.Tensor) : input data tensor
        eps (float) : a small number used for numerial stablility.
        biased (bool): Whether to use the biased estimate of the covariance matrix
        return_inv (bool): Whether to return the inverse transform matrix
    
    shape:

    returns:
        Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        A tuple containing the ZCA matrix and the mean vector, and if return_inv = True, the 
        inverse whitening matrix is retured as well.

    """

    N: int = inp.size()[0]
    num_features: int = reduce(lambda a,b: a*b, inp.size()[1::])

    mean: torch.Tensor = torch.mean(inp, dim=0, keepdim=True)
    mean = mean.view((1,num_features))
    
    inp_center_flat: torch.Tensor = inp.view((N, num_features)) - mean

    cov = inp_center_flat.t().mm(inp_center_flat)

    if biased:
        cov = cov/float(N)
    else:
        cov = cov/float(N-1)

    U, S, _ = torch.svd(cov)

    S = S.view(-1,1)
    S_inv_root: torch.Tensor = torch.rsqrt(S + eps)
    S_inv_root = S_inv_root

    T: torch.Tensor = U.mm(S_inv_root*U.t())

    if return_inv:
        T_inv = U.t().mm(S * U)
        return T, T_inv, mean
    else:
        return T, mean
