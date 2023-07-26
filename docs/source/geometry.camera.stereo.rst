Stereo Camera
-------------

.. currentmodule:: kornia.geometry.camera.stereo

In this module we provide the :class:`StereoCamera` that contains functionality for working with a horizontal stereo camera setup.

The horizontal stereo camera setup is assumed to be calibrated and rectified such that the setup can be described by two camera matrices:

The *left rectified camera matrix*:

.. math::
    P_0 = \begin{bmatrix}
    fx & 0  & cx & 0 \\
    0  & fy & cy & 0 \\
    0  & 0  & 1  & 0
    \end{bmatrix}

The *right rectified camera matrix*:

.. math::
    P_1 = \begin{bmatrix}
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

This is done by forming the :math:`Q` matrix.

Using the pinhole camera model to project :math:`[X Y Z 1]` in world coordinates to :math:`uv` pixels in the left and right camera frame respectively:

.. math::

    \begin{bmatrix}
    u \\
    v \\
    1
    \end{bmatrix} = P_0 *
    \begin{bmatrix}
    X \\
    Y \\
    Z \\
    1
    \end{bmatrix} \\
    \begin{bmatrix}
    u-d \\
    v \\
    1
    \end{bmatrix} = P_1 *
    \begin{bmatrix}
    X \\
    Y \\
    Z \\
    1
    \end{bmatrix}

Where :math:`d` is the disparity between pixels in left and right image.

Combining these two expressions let us write it as one matrix multiplication

.. math::
    \begin{bmatrix}
    u \\
    v \\
    u-d \\
    1
    \end{bmatrix} =
    \begin{bmatrix}
    fx & 0 & cx_{left} & 0 \\
    0  & fy & cy & 0 \\
    fx & 0 & cx_{right} & fx * tx \\
    0  & 0 & 1 & 0
    \end{bmatrix}
    \begin{bmatrix}
    X \\
    Y \\
    Z \\
    1
    \end{bmatrix}

Now subtract the first from the third row and invert the expression and you'll get:

.. math::
    \begin{bmatrix}
    u \\
    v \\
    d \\
    1
    \end{bmatrix} =
    \begin{bmatrix}
    fy * tx & 0       & 0   & -fy * cx * tx \\
    0       & fx * tx & 0   & -fx * cy * tx \\
    0       & 0       & 0   & fx * fy * tx  \\
    0       & 0       & -fy & fy * (cx_{left} -cx_{right})
    \end{bmatrix}
    \begin{bmatrix}
    X \\
    Y \\
    Z \\
    1
    \end{bmatrix}

Where :math:`Q` is

.. math::

    Q = \begin{bmatrix}
    fy * tx & 0       & 0   & -fy * cx * tx \\
    0       & fx * tx & 0   & -fx * cy * tx \\
    0       & 0       & 0   & fx * fy * tx  \\
    0       & 0       & -fy & fy * (cx_{left} -cx_{right})
    \end{bmatrix}

Notice here that the x-coordinate for the principal point in the left and right camera :math:`cx` might differ, which is being taken into account here.

Assuming :math:`fx = fy` you can further reduce this to:

.. math::
    Q = \begin{bmatrix}
    1 & 0       & 0   & -cx \\
    0       & 1 & 0   & -cy \\
    0       & 0       & 0   & fx  \\
    0       & 0       & -1/tx & (cx_{left} -cx_{right} / tx)
    \end{bmatrix}

But we'll use the general :math:`Q` matrix.

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
