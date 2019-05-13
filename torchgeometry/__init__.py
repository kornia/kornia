# Make sure that torchgeometry is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)
import sys
if sys.version_info < (3, 6, 0):
    raise RuntimeError("Pytorch Geometry requires Python 3.6.0 or later")

from .version import __version__

from torchgeometry import geometry
from torchgeometry import color
from torchgeometry import feature
from torchgeometry import filters
from torchgeometry import losses
from torchgeometry import contrib
from torchgeometry import metrics
from torchgeometry import utils

# Exposes ``torchgeometry.geometry`` package to top level
from torchgeometry.geometry import *
