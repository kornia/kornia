from torchgeometry import core
from torchgeometry import image
from torchgeometry import losses
from torchgeometry import contrib
from torchgeometry import utils

# Exposes ``torchgeometry.core`` package to top level
from .core.homography_warper import HomographyWarper, homography_warp
from .core.depth_warper import DepthWarper, depth_warp
from .core.pinhole import *
from .core.conversions import *
from .core.imgwarp import *
from .core.transformations import *


# TODO(edgar): move to a separated file
__version__ = '0.1.2rc1'  # the current version of the lib
