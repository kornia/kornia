from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.feature import HessianResp
from kornia.feature.laf import (
    denormalize_LAF,
    normalize_LAF
    )

from kornia import ScapePyramid

class PassLAF(nn.Module):
    def forward(self, laf, img):
        return laf
