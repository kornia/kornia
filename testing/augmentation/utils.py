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

import copy

import torch

from testing.base import assert_close


def oversized_random_crop_params(crop, batch_shape):
    """Return the parameters ``crop.forward_parameters(batch_shape)`` gave before #4414 for an oversized crop.

    ``RandomCrop`` now rejects a ``size`` larger than its padded canvas when ``pad_if_needed=False``, but a replay of
    such parameters still reaches its oversized-crop matrix scaling. A twin whose extra bottom and right padding makes
    its canvas exactly the crop size on an oversized axis draws the same crop box, because the generator places a box
    at offset 0 on an axis it spans, either way, from the same random draws. The twin's padding fields are then
    replaced by the crop's own.
    """
    padding = crop.compute_padding(batch_shape)  # left, right, top, bottom
    canvas = (batch_shape[-2] + padding[2] + padding[3], batch_shape[-1] + padding[0] + padding[1])
    extra_h, extra_w = (max(s - c, 0) for s, c in zip(crop.flags["size"], canvas))
    twin = copy.deepcopy(crop)
    twin.flags["padding"] = (padding[0], padding[2], padding[1] + extra_w, padding[3] + extra_h)
    params = twin.forward_parameters(batch_shape)
    params["padding_size"] = params["padding_size"].new_tensor(padding).expand(batch_shape[0], -1)
    params["forward_input_shape"] = params["forward_input_shape"].new_tensor((*batch_shape[:-2], *canvas))
    params["input_size"] = params["input_size"].new_tensor(canvas).expand(batch_shape[0], -1)
    return params


def reproducibility_test(input, seq):
    """Any tests failed here indicate the output cannot be reproduced by the same params."""
    if isinstance(input, (tuple, list)):
        output_1 = seq(*input)
        output_2 = seq(*input, params=seq._params)
    else:
        output_1 = seq(input)
        output_2 = seq(input, params=seq._params)

    if isinstance(output_1, (tuple, list)) and isinstance(output_2, (tuple, list)):
        [
            assert_close(o1, o2)
            for o1, o2 in zip(output_1, output_2)
            if isinstance(o1, (torch.Tensor,)) and isinstance(o2, (torch.Tensor,))
        ]
    elif isinstance(output_1, (tuple, list)) and isinstance(output_2, (torch.Tensor,)):
        assert_close(output_1[0], output_2)
    elif isinstance(output_2, (tuple, list)) and isinstance(output_1, (torch.Tensor,)):
        assert_close(output_1, output_2[0])
    elif isinstance(output_2, (torch.Tensor,)) and isinstance(output_1, (torch.Tensor,)):
        assert_close(output_1, output_2, msg=f"{seq._params}")
    else:
        raise AssertionError(f"cannot compare {type(output_1)} and {type(output_2)}")
