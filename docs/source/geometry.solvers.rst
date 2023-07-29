kornia.geometry.solvers
=======================

.. currentmodule:: kornia.geometry.solvers

Module containing various geometrical solvers/optimizers.

Polynomial Solvers
------------------

.. autofunction:: solve_quadratic
.. note::
   In cases where a quadratic polynomial has only one real root, the output will be in the format [real_root, 0]. And for the
   complex roots should be represented as 0. This is done to maintain a consistent output shape for all cases.

.. autofunction:: solve_cubic
.. note::
   In cases where a cubic polynomial has only one or two real roots, the output for the non-real roots should be represented as 0. Thus, the output for a single real root should be in the format [real_root, 0, 0], and for two real roots, it should be [real_root_1, real_root_2, 0].
