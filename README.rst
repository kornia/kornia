torchgeometry
=============

The torchgeometry package consists of functionalities to make Geometric Computer Vision differentiable.

Quick Usage
===========

.. code:: python

 import torch
 import torchgeometry as dgm  # differential geometry

 x_rad = dgm.pi * torch.rand(1, 3, 3)
 x_deg = dgm.rad2deg(x_rad)

 torch.allclose(x_rad, dgm.deg2rad(x_deg))  # True


Installation
============

From source:

.. code:: bash

    python setup.py install

Testing
=======

.. code:: bash

    python setup.py test

Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

