# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch

from .base import MixAugmentationBaseV2


class PatchMix(MixAugmentationBaseV2):
    def __init__(self, alpha=1.0, patch_size=16, p=1.0, same_on_batch=False, keepdim=False):
        super().__init__(p=p, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.alpha = alpha
        self.patch_size = patch_size

    def generate_parameters(self, batch_shape):
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_shape[0],))
        # ensure lambda is on the module RNG device and dtype when available
        dev = None
        dt = None
        try:
            dev = self.device
            dt = self.dtype
        except AttributeError:
            dev = None
            dt = None
        if dev is not None or dt is not None:
            to_kwargs = {}
            if dev is not None:
                to_kwargs["device"] = dev
            if dt is not None:
                to_kwargs["dtype"] = dt
            lam = lam.to(**to_kwargs)
        return {"lam": lam}

    def apply_transform(self, input, params, extra_args):
        B, _, H, W = input.shape
        lam = params["lam"].to(input.device)
        idx = torch.randperm(B, device=input.device)
        out = input.clone()
        for i in range(B):
            # Random patch coordinates
            y = int(torch.randint(0, H - self.patch_size + 1, (), device=input.device).item())
            x = int(torch.randint(0, W - self.patch_size + 1, (), device=input.device).item())
            out[i, :, y : y + self.patch_size, x : x + self.patch_size] = input[
                idx[i], :, y : y + self.patch_size, x : x + self.patch_size
            ]
        return out, idx, lam
