torchgeometry
=============

.. image:: https://travis-ci.com/arraiy/torchgeometry.svg?branch=master
    :target: https://travis-ci.com/arraiy/torchgeometry

What is torchgeometry?
===========
The torchgeometry package consists of a set of routines and differentiable modules to solve generic geometry problems. At its core, the package uses PyTorch as its main backend both for efficiency and to take advantage of autograd to define and compute the gradient of complex functions. In this first version, we provide different functions designed mainly for computer vision applications, such as image and tensors warping modules which rely on the epipolar geometry theory. The roadmap will include adding more and more functionality so that developers in the short term can use the package for the purpose of optimizing their loss functions to solve geometry problems.

Version v0.1.0 focuses on Image and tensor warping functions such as:

* Calibration
* Epipolar geometry
* Homography
* Depth

Quick Usage
===========

.. code:: python

 import torch
 import torchgeometry as dgm  # differential geometry

 x_rad = dgm.pi * torch.rand(1, 3, 3)
 x_deg = dgm.rad2deg(x_rad)

 torch.allclose(x_rad, dgm.deg2rad(x_deg))  # True
 
-------------------------------------------------------

.. code:: python

 import torch
 import torchgeometry as dgm  # differential geometry

 img_ref = torch.rand(1, 3, 32, 32)  # NxCxHxW
 dst_homo_ref = torch.eye(1) + eps   # Nx3x3

 warper = dgm.HomographyWarper(32, 32)
 img_ref_to_dst = warper(img_ref, dst_homo_ref)  # NxCxHxW
 
 
Installation
============

From source:

.. code:: bash

    python setup.py install

Testing
=======

.. code:: bash

    python setup.py test

Cite
============
If you are using DGM in your research-related documents, it is recommended that you cite DGM poster. 


Future work
============
The `roadmap <https://github.com/arraiy/torchgeometry/issues/1>`_ will add more functions to allow developers to solve geometric problems.


Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.
