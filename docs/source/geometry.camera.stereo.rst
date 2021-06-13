Stereo Camera
-------------

.. currentmodule:: kornia.geometry.camera.stereo

In this module we provide the :class:`StereoCamera` that contains functionality for working with a horizontal stereo camera setup.

The horizontal stereo camera setup is assumed to be calibrated and rectified such that the setup can be described by two camera matrices:

The *left rectified camera matrix*:

.. math::
    \begin{bmatrix}
    fx & 0  & cx & 0 \\
    0  & fy & cy & 0 \\
    0  & 0  & 1  & 0
    \end{bmatrix}

The *right rectified camera matrix*:

.. math::
    \begin{bmatrix}
    fx & 0  & cx & tx * fx \\
    0  & fy & cy & 0       \\
    0  & 0  & 1  & 0
    \end{bmatrix}

where:
    * :math:`fx` is the focal length in the x-direction in pixels.
    * :math:`fy` is the focal length in the y-direction in pixels.
    * :math:`cx` is the x-coordinate of the principal point in pixels.
    * :math:`cy` is the y-coordinate of the principal point in pixels.
    * :math:`tx` is the horizontal baseline in metric units.

These camera matrices are obtained by calibrating your stereo camera setup which can be done `in OpenCV <https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5>`_.

The :class:`StereoCamera` allows you to convert disparity maps to the real world 3D geometry represented by a point cloud.

This is done by forming the :math:`Q` matrix:

.. math::

    Q = \begin{bmatrix}
    fy * tx & 0       & 0   & -fy * cx * tx \\
    0       & fx * tx & 0   & -fx * cy * tx \\
    0       & 0       & 0   & fx * fy * tx  \\
    0       & 0       & -fy & fy * (cx_{left} -cx_{right})
    \end{bmatrix}

Notice here that the x-coordinate for the principal point in the left and right camera :math:`cx` might differ, which is being taken into account here.

Using the :math:`Q` matrix we can obtain the 3D points by:

.. math::
    \begin{bmatrix}
    X \\
    Y \\
    Z \\
    W
    \end{bmatrix} = Q *
    \begin{bmatrix}
    u \\
    v \\
    disparity(y, v) \\
    z
    \end{bmatrix}

.. autoclass:: StereoCamera
    :members:

    .. automethod:: __init__

.. autofunction:: reproject_disparity_to_3D
