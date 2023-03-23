from typing import List, Optional

import torch
import torch.nn.functional as F

from kornia.core import Tensor

from .structs import Keypoints


def select_on_last(values: Tensor, indices: Tensor) -> Tensor:
    '''
    WARNING: this may be reinventing the wheel, but I don't know how to do
    it otherwise with PyTorch.

    This function uses an array of linear indices `indices` between [0, T] to
    index into `values` which has equal shape as `indices` and then one extra
    dimension of size T.
    '''
    return torch.gather(values, -1, indices[..., None]).squeeze(-1)


def nms(signal: Tensor, window_size=5, cutoff=0.0) -> Tensor:
    if window_size % 2 != 1:
        raise ValueError(f'window_size has to be odd, got {window_size}')

    _, ixs = F.max_pool2d(signal, kernel_size=window_size, stride=1, padding=window_size // 2, return_indices=True)

    h, w = signal.shape[1:]
    coords = torch.arange(h * w, device=signal.device).reshape(1, h, w)
    nms = ixs == coords

    if cutoff is None:
        return nms
    else:
        return nms & (signal > cutoff)


class Detector:
    def __init__(self, window=8):
        self.window = window

    def nms(self, heatmap: Tensor, n: Optional[int] = None, **kwargs) -> List[Keypoints]:
        """Inference-time nms-based detection protocol."""
        heatmap = heatmap.squeeze(1)
        nmsed = nms(heatmap, **kwargs)

        keypoints = []
        for b in range(heatmap.shape[0]):
            yx = nmsed[b].nonzero(as_tuple=False)
            logp = heatmap[b][nmsed[b]]
            xy = yx.flip((1,))

            if n is not None:
                n_ = min(n + 1, logp.numel())
                # torch.kthvalue picks in ascending order and we want to pick in
                # descending order, so we pick n-th smallest among -logp to get
                # -threshold
                minus_threshold, _indices = torch.kthvalue(-logp, n_)
                mask = logp > -minus_threshold

                xy = xy[mask]
                logp = logp[mask]

            keypoints.append(Keypoints(xy, logp))

        return keypoints
