Pinhole
--------

.. currentmodule:: torchgeometry

.. note::
    The pinhole model is represented in a single vector as follows:

    .. math::
        pinhole = (f_x, f_y, c_x, c_y, height, width, r_x, r_y, r_z, t_x, t_y, t_z)
 
    where:
        :math:`(r_x, r_y, r_z)` is the rotation vector in angle-axis convention.

        :math:`(t_x, t_y, t_z)` is the translation vector.

.. autofunction:: inverse_pose
.. autofunction:: pinhole_matrix
.. autofunction:: inverse_pinhole_matrix
.. autofunction:: scale_pinhole
.. autofunction:: homography_i_H_ref

.. autoclass:: InversePose
.. autoclass:: PinholeMatrix
.. autoclass:: InversePinholeMatrix
