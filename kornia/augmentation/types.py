from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch

TupleFloat = Tuple[float, float]
UnionFloat = Union[float, TupleFloat]
UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]
BoarderUnionType = Union[int, Tuple[int, int], Tuple[int, int, int, int]]
