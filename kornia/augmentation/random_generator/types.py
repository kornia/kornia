from typing import Dict, Optional

import torch
try:
    from typing import TypedDict  # type: ignore  # >=3.8  
except ImportError:
    from mypy_extensions import TypedDict  # type: ignore  # <=3.7


class AugParamDict(TypedDict, total=False):  # type: ignore
    batch_prob: torch.Tensor
    params: Dict[str, torch.Tensor] = {}
    flags: Dict[str, torch.Tensor] = {}
