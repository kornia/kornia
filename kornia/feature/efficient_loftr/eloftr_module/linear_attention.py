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

"""Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention".

Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

from typing import List, Optional

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor

if hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
    from torch.backends.cuda import sdp_kernel
else:
    FLASH_AVAILABLE = False


def crop_feature(query: Tensor, key: Tensor, value: Tensor, x_mask: Tensor, source_mask: Tensor) -> List[Tensor]:
    """Cropping features."""
    mask_h0, mask_w0, mask_h1, mask_w1 = (
        x_mask[0].sum(-2)[0],
        x_mask[0].sum(-1)[0],
        source_mask[0].sum(-2)[0],
        source_mask[0].sum(-1)[0],
    )
    query = query[:, :mask_h0, :mask_w0, :]
    key = key[:, :mask_h1, :mask_w1, :]
    value = value[:, :mask_h1, :mask_w1, :]
    return query, key, value, mask_h0, mask_w0


def pad_feature(m: Tensor, mask_h0: Tensor, mask_w0: Tensor, x_mask: Tensor) -> Tensor:
    """Padding features."""
    bs, L, H, D = m.size()
    m = m.view(bs, mask_h0, mask_w0, H, D)
    if mask_h0 != x_mask.size(-2):
        m = torch.cat(
            [
                m,
                torch.zeros(
                    m.size(0), x_mask.size(-2) - mask_h0, x_mask.size(-1), H, D, device=m.device, dtype=m.dtype
                ),
            ],
            dim=1,
        )
    elif mask_w0 != x_mask.size(-1):
        m = torch.cat(
            [
                m,
                torch.zeros(
                    m.size(0), x_mask.size(-2), x_mask.size(-1) - mask_w0, H, D, device=m.device, dtype=m.dtype
                ),
            ],
            dim=2,
        )
    return m


class Attention(Module):
    def __init__(self, no_flash: bool = False, nhead: int = 8, dim: int = 256, fp32: bool = False) -> None:
        super().__init__()
        self.flash = FLASH_AVAILABLE and not no_flash
        self.nhead = nhead
        self.dim = dim
        self.fp32 = fp32

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # assert q_mask is None and kv_mask is None, "Not support generalized attention mask yet."
        if q_mask is not None:
            raise AssertionError(q_mask)
        if kv_mask is not None:
            raise AssertionError(kv_mask)
        if self.flash and not self.fp32:
            args = [x.contiguous() for x in [query, key, value]]
            with sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                out = F.scaled_dot_product_attention(*args)
        elif self.flash:
            args = [x.contiguous() for x in [query, key, value]]
            out = F.scaled_dot_product_attention(*args)
        else:
            QK = torch.einsum("nlhd,nshd->nlsh", query, key)

            # Compute the attention and the weighted average
            softmax_temp = 1.0 / query.size(3) ** 0.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)

            out = torch.einsum("nlsh,nshd->nlhd", A, value)
        return out

    def _forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if q_mask is not None and kv_mask is not None:
            query, key, value, mask_h0, mask_w0 = crop_feature(query, key, value, q_mask, kv_mask)

        if self.flash:
            # query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n nhead (h w) d',
            # nhead=self.nhead, d=self.dim), [query, key, value])
            query, key, value = [
                x.view(x.size(0), x.size(1), x.size(2), self.nhead, self.dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(x.size(0), self.nhead, -1, self.dim)
                for x in (query, key, value)
            ]
        else:
            # query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n (h w) nhead d',
            # nhead=self.nhead, d=self.dim), [query, key, value])
            query, key, value = [x.reshape(x.size(0), -1, self.nhead, self.dim) for x in (query, key, value)]

        m = self.attention(query, key, value, q_mask=None, kv_mask=None)

        if self.flash:
            # m = rearrange(m, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)
            # n0, nhead, L, d = m.shape
            m = m.permute(0, 2, 1, 3)

        if q_mask is not None:
            m = pad_feature(m, mask_h0, mask_w0, q_mask)

        return m

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-head scaled dot-product attention, a.k.a full attention.

        Args:
            query: [N, H, L, D]
            key: [N, H, S, D]
            value: [N, H, S, D]
            q_mask: [N, L]
            kv_mask: [N, S]

        Returns:
            queried_values: (N, L, H, D)
        """
        bs = query.size(0)
        if bs == 1 or q_mask is None:
            m = self._forward(query, key, value, q_mask=q_mask, kv_mask=kv_mask)
        else:  # for faster training with padding mask while batch size > 1
            m_list = []
            for i in range(bs):
                m_list.append(
                    self._forward(
                        query[i : i + 1],
                        key[i : i + 1],
                        value[i : i + 1],
                        q_mask=q_mask[i : i + 1],
                        kv_mask=kv_mask[i : i + 1],
                    )
                )
            m = torch.cat(m_list, dim=0)
        return m
