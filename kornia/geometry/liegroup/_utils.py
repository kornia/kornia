from kornia.core import Tensor
from kornia.testing import KORNIA_CHECK


def squared_norm(x, y=None) -> Tensor:
    return _batched_squared_norm(x, y)


def _batched_squared_norm(x, y=None):
    if y is None:
        y = x
    KORNIA_CHECK(x.shape == y.shape)
    return (x[..., None, :] @ y[..., :, None])[..., 0]
