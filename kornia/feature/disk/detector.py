from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Bernoulli, Categorical

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


def point_distribution(logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements the categorical proposal -> Bernoulli acceptance sampling scheme.

    Given a tensor of logits, performs samples on the last dimension, returning     a) the proposals     b) a binary
    mask indicating which ones were accepted     c) the logp-probability of (proposal and acceptance decision)
    """

    proposal_dist = Categorical(logits=logits)
    proposals = proposal_dist.sample()
    proposal_logp = proposal_dist.log_prob(proposals)

    accept_logits = select_on_last(logits, proposals).squeeze(-1)

    accept_dist = Bernoulli(logits=accept_logits)
    accept_samples = accept_dist.sample()
    accept_logp = accept_dist.log_prob(accept_samples)
    accept_mask = accept_samples == 1.0

    logp = proposal_logp + accept_logp

    return proposals, accept_mask, logp


def nms(signal: Tensor, window_size=5, cutoff=0.0) -> Tensor:
    if window_size % 2 != 1:
        raise ValueError(f'window_size has to be odd, got {window_size}')

    _, ixs = F.max_pool2d(signal, kernel_size=window_size, stride=1, padding=window_size // 2, return_indices=True)

    # FIXME UPSTREAM: a workaround wrong shape of `ixs` until
    # https://github.com/pytorch/pytorch/issues/38986
    # is fixed
    if len(ixs.shape) == 4:
        assert ixs.shape[0] == 1
        ixs = ixs.squeeze(0)

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

    def _tile(self, heatmap: Tensor) -> Tensor:
        """Divides the heatmap `heatmap` into tiles of size (v, v) where v==self.window. The tiles are flattened,
        resulting in the last.

        dimension of the output T == v * v.
        """
        v = self.window
        b, c, h, w = heatmap.shape

        assert heatmap.shape[2] % v == 0
        assert heatmap.shape[3] % v == 0

        return heatmap.unfold(2, v, v).unfold(3, v, v).reshape(b, c, h // v, w // v, v * v)

    def sample(self, heatmap: Tensor) -> np.ndarray:
        """Implements the training-time grid-based sampling protocol."""
        v = self.window
        dev = heatmap.device
        B, _, H, W = heatmap.shape

        assert H % v == 0
        assert W % v == 0

        # tile the heatmap into [window x window] tiles and pass it to
        # the categorical distribution.
        heatmap_tiled = self._tile(heatmap).squeeze(1)
        proposals, accept_mask, logp = point_distribution(heatmap_tiled)

        # create a grid of xy coordinates and tile it as well
        cgrid = torch.stack(
            torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')[::-1], dim=0
        ).unsqueeze(0)

        # extract xy coordinates from cgrid according to indices sampled
        # before
        xys = select_on_last(
            self._tile(cgrid).repeat(B, 1, 1, 1, 1),
            # unsqueeze and repeat on the (xy) dimension to grab
            # both components from the grid
            proposals.unsqueeze(1).repeat(1, 2, 1, 1),
        ).permute(
            0, 2, 3, 1
        )  # -> bhw2

        keypoints = []
        for i in range(B):
            mask = accept_mask[i]
            keypoints.append(Keypoints(xys[i][mask], logp[i][mask]))

        return np.array(keypoints, dtype=object)

    def nms(self, heatmap: Tensor, n: Optional[int] = None, **kwargs) -> np.ndarray:
        """Inference-time nms-based detection protocol."""
        heatmap = heatmap.squeeze(1)
        nmsed = nms(heatmap, **kwargs)

        keypoints = []
        for b in range(heatmap.shape[0]):
            yx = nmsed[b].nonzero(as_tuple=False)
            logp = heatmap[b][nmsed[b]]
            xy = torch.flip(yx, (1,))

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

        return np.array(keypoints, dtype=object)
