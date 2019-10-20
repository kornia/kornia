Frequently Asked Questions
==========================

This document contains frequently asked questions.

How to Install
--------------

See `Installation <https://kornia.readthedocs.io/en/latest/installation.html/>`_

How to Contribute
-----------------

See `How to Contribute <https://kornia.readthedocs.io/en/latest/contribute.html/>`_

Kornia relation to Pytorch Geometry/Geometric
---------------------------------------------

This project started as a small differentiable geometric computer
vision package called `PyTorch Geometry <https://pypi.org/project/torchgeometry>`_
released during the PyTorch devcon 2018 (see the presented 
`poster <https://drive.google.com/file/d/1xiao1Xj9WzjJ08YY_nYwsthE-wxfyfhG/view?usp=sharing>`_).
The project evolved to a more generic computer vision library and due to the naming
conflict between `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest>`_
we decided to rename the whole package and focus to more generic vision functionalities.

Kornia relation to Other Computer Vision Projects
-------------------------------------------------

The project mimics some of the functionalities found in OpenCV. Eventhough
the project is backed up by the `OpenCV.org <www.opencv.org/>`_, there is no
intention at all to merge in any form both projects. *Kornia* is ment to
provide differentiable operators to train nets, while *OpenCV* scope is inference.
