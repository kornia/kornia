import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import kornia

TupleTensor = Tuple[torch.Tensor, torch.Tensor]
