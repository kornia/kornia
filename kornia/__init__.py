# Make sure that kornia is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)
import sys

if sys.version_info < (3, 6, 0):
    raise RuntimeError("Kornia requires Python 3.6.0 or later")

try:
    from .version import __version__  # nopa: 401
except ImportError:
    pass

# NOTE: kornia filters and geometry must go first since are the core of the library
# and by changing the import order you might get into a circular dependencies issue.
from . import filters
from . import geometry

# import the other modules for convenience
from . import (
    augmentation,
    color,
    contrib,
    enhance,
    feature,
    losses,
    metrics,
    morphology,
    utils,
    x,
)
# NOTE: we are going to expose to top level very few things
from kornia.constants import pi
from kornia.testing import xla_is_available
from kornia.utils import (
    eye_like,
    vec_like,
    create_meshgrid,
    image_to_tensor,
    tensor_to_image,
)
