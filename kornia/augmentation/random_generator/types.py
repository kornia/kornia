from typing import Dict, Optional

import torch
try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7


class AugParamDict(TypedDict, total=False):
    batch_prob: torch.Tensor
    params: Dict[str, torch.Tensor] = {}
    flags: Dict[str, torch.Tensor] = {}
