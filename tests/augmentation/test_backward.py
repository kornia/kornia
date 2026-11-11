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

import pytest
import torch
from torch import nn

import kornia.augmentation as K

from testing.base import supports_bilinear_2d_grid_sample_backward

# Every tensor form a learnable range argument accepts, one argument at a time with the others left as Python numbers:
# (augmentation, argument, value, volumetric). Scales stay near 1, since ``(0.1, 0.2)`` shrinks a depth-4 volume below
# one voxel and the gradient vanishes. ``TestLearnableRangeCopies`` and ``TestLearnableRangeFollowsItsValue`` in
# ``test_base.py`` train one form of each family; RandomResizedCrop and RandomErasing draw their ranges without a
# gradient and are covered there.
_COLOR_FORMS = [
    (name, arg, value, False)
    for name in ("ColorJiggle", "ColorJitter")
    for arg, values in (
        ("brightness", (torch.tensor(0.8), torch.tensor([0.8, 1.2]))),
        ("contrast", (torch.tensor(0.8), torch.tensor([0.8, 1.2]))),
        ("saturation", (torch.tensor(0.8), torch.tensor([0.8, 1.2]))),
        ("hue", (torch.tensor(0.1), torch.tensor([-0.1, 0.1]))),
    )
    for value in values
]
_LEARNABLE_FORMS = [
    *_COLOR_FORMS,
    ("RandomAffine", "degrees", torch.tensor(10.0), False),
    ("RandomAffine", "degrees", torch.tensor([10, 20]), False),
    ("RandomAffine", "translate", torch.tensor([0.1, 0.2]), False),
    ("RandomAffine", "scale", torch.tensor([0.9, 1.1]), False),
    ("RandomAffine", "scale", torch.tensor([0.9, 1.1, 0.9, 1.1]), False),
    ("RandomAffine", "shear", torch.tensor([10, 20]), False),
    ("RandomAffine", "shear", torch.tensor([10, 20, 30, 40]), False),
    ("RandomRotation", "degrees", torch.tensor(10.0), False),
    ("RandomRotation", "degrees", torch.tensor([10, 20]), False),
    ("RandomPerspective", "distortion_scale", torch.tensor(0.5), False),
    ("RandomMotionBlur", "angle", torch.tensor([-20.0, 20.0]), False),
    ("RandomMotionBlur", "direction", torch.tensor([-0.5, 0.5]), False),
    ("RandomSharpness", "sharpness", torch.tensor([0, 0.5]), False),
    ("RandomAffine3D", "degrees", torch.tensor(10.0), True),
    ("RandomAffine3D", "degrees", torch.tensor([10.0, 20.0]), True),
    ("RandomAffine3D", "degrees", torch.tensor([10, 20, 30]), True),
    ("RandomAffine3D", "degrees", torch.tensor([(10, 20), (10, 20), (10, 20)]), True),
    ("RandomAffine3D", "translate", torch.tensor([0.1, 0.2, 0.3]), True),
    ("RandomAffine3D", "scale", torch.tensor([0.9, 1.1]), True),
    ("RandomAffine3D", "scale", torch.tensor([(0.9, 1.1), (0.9, 1.1), (0.9, 1.1)]), True),
    ("RandomAffine3D", "shears", torch.tensor(10), True),
    ("RandomAffine3D", "shears", torch.tensor([10, 20]), True),
    ("RandomAffine3D", "shears", torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]), True),
    ("RandomAffine3D", "shears", torch.tensor([(-10.0, 10.0)] * 6), True),
]

# The arguments each augmentation needs besides the learnable one.
_FIXED_ARGS = {
    "ColorJiggle": {"brightness": 0.8, "contrast": 0.8, "saturation": 0.8, "hue": 0.1},
    "ColorJitter": {"brightness": 0.8, "contrast": 0.8, "saturation": 0.8, "hue": 0.1},
    "RandomAffine": {"degrees": 10.0},
    "RandomRotation": {"degrees": 10.0},
    "RandomPerspective": {},
    # ``nearest``, the default, copies kernel taps, which has no gradient with respect to the angle.
    "RandomMotionBlur": {"kernel_size": 3, "angle": 20.0, "direction": 0.5, "resample": "bilinear"},
    "RandomSharpness": {},
    "RandomAffine3D": {"degrees": 10.0},
}


def _form_id(form):
    name, arg, value, _ = form
    return f"{name}-{arg}-{'x'.join(map(str, value.shape)) or '0d'}-{str(value.dtype).removeprefix('torch.')}"


class TestLearnableRangeForms:
    @pytest.mark.parametrize("form", _LEARNABLE_FORMS, ids=_form_id)
    def test_gradient_reaches_the_range(self, form, device, dtype):
        name, arg, value, volumetric = form
        if name in ("RandomAffine", "RandomRotation") and not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip("grid_sample backward is unavailable for this device/dtype")
        # ``clone``: on a matching device and dtype ``to`` returns the table entry itself, shared by every test.
        param = nn.Parameter(value.clone().to(device=device, dtype=dtype))
        aug = getattr(K, name)(**{**_FIXED_ARGS[name], arg: param}, p=1.0)
        assert any(p is param for p in aug.parameters())

        torch.manual_seed(0)
        shape = (2, 3, 4, 10, 10) if volumetric else (2, 3, 10, 10)
        image = torch.randint(255, shape, device=device).to(dtype) / 255.0
        # A target outside the image range gives every draw a non-zero loss gradient.
        nn.functional.mse_loss(aug(image), torch.full_like(image, 2.0)).backward()
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert torch.count_nonzero(param.grad) > 0
