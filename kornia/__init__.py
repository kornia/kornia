# NOTE: kornia filters and geometry must go first since are the core of the library
# and by changing the import order you might get into a circular dependencies issue.
from . import filters
from . import geometry
from . import grad_estimator

# import the other modules for convenience
from . import augmentation, color, contrib, core, enhance, feature, io, losses, metrics, morphology, tracking, utils, x

# NOTE: we are going to expose to top level very few things
from kornia.constants import pi
from kornia.testing import xla_is_available
from kornia.utils import eye_like, vec_like, create_meshgrid, image_to_tensor, tensor_to_image

# Version variable
import sys

if sys.version_info >= (3, 8):  # pragma: >=3.8 cover
    import importlib.metadata as importlib_metadata
else:  # pragma: <3.8 cover
    import importlib_metadata

__version__ = importlib_metadata.version('kornia')
