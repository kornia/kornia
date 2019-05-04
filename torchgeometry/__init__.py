# Make sure that torchgeometry is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)
import sys
if sys.version_info < (3, 6, 0):
    raise RuntimeError("Pytorch Geometry requires Python 3.6.0 or later")

from .version import __version__

from torchgeometry import core
from torchgeometry import image
from torchgeometry import losses
from torchgeometry import contrib
from torchgeometry import utils
from torchgeometry import metrics

# Exposes ``torchgeometry.core`` package to top level
from .core.homography_warper import HomographyWarper, homography_warp
from .core.depth_warper import DepthWarper, depth_warp
from .core.pinhole import *
from .core.conversions import *
from .core.imgwarp import *
from .core.transformations import *
from .core.affine import (
    affine, rotate, translate, scale, shear, Rotate, Translate, Scale, Shear
)
from .core.crop import center_crop
