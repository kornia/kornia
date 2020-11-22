from typing import Tuple, Union, Optional

import torch
import torch.nn as nn

import kornia


class BlobDoG(nn.Module):
    r"""nn.Module that calculates Difference-of-Gaussians blobs
    See :func:`~kornia.feature.dog_response` for details.
    """

    def __init__(self) -> None:
        super(BlobDoG, self).__init__()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def forward(self, input: torch.Tensor,  # type: ignore
                sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        return kornia.feature.dog_response(input)  # type: ignore


class CornerHarris(nn.Module):
    r"""nn.Module that calculates Harris corners
    See :func:`~kornia.feature.harris_response` for details.
    """

    def __init__(self, k: Union[float, torch.Tensor],
                 grads_mode='sobel') -> None:
        super(CornerHarris, self).__init__()
        if type(k) is float:
            self.register_buffer('k', torch.tensor(k))
        else:
            self.register_buffer('k', k)  # type: ignore
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(k=' + str(self.k) + ', ' +\
            'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: torch.Tensor,  # type: ignore
                sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        return kornia.feature.harris_response(input, self.k, self.grads_mode, sigmas)  # type: ignore


class CornerGFTT(nn.Module):
    r"""nn.Module that calculates Shi-Tomasi corners
    See :func:`~kornia.feature.gfft_response` for details.
    """

    def __init__(self, grads_mode='sobel') -> None:
        super(CornerGFTT, self).__init__()
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: torch.Tensor,  # type: ignore
                sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        return kornia.feature.gftt_response(input, self.grads_mode, sigmas)


class BlobHessian(nn.Module):
    r"""nn.Module that calculates Hessian blobs
    See :func:`~kornia.feature.hessian_response` for details.
    """

    def __init__(self, grads_mode='sobel') -> None:
        super(BlobHessian, self).__init__()
        self.grads_mode: str = grads_mode

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: torch.Tensor,  # type: ignore
                sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
        return kornia.feature.hessian_response(input, self.grads_mode, sigmas)
