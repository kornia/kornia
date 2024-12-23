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

from typing import Any, Callable

from kornia.core import Module, Tensor


class Lambda(Module):
    """Applies user-defined lambda as a transform.

    Args:
        func: Callable function.

    Returns:
        The output of the user-defined lambda.

    Example:
        >>> import kornia
        >>> x = torch.rand(1, 3, 5, 5)
        >>> f = Lambda(lambda x: kornia.color.rgb_to_grayscale(x))
        >>> f(x).shape
        torch.Size([1, 1, 5, 5])

    """

    def __init__(self, func: Callable[..., Tensor]) -> None:
        super().__init__()
        if not callable(func):
            raise TypeError(f"Argument lambd should be callable, got {type(func).__name__!r}")

        self.func = func

    def forward(self, img: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return self.func(img, *args, **kwargs)
