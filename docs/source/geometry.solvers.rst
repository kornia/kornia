kornia.geometry.solvers
=======================

.. currentmodule:: kornia.geometry.solvers

Module containing various geometrical solvers/optimizers.

Polynomial Solvers
------------------

.. autofunction:: solve_quadratic

Finds the real roots of a quadratic equation

.. math:: coeffs[0]x^2 + coeffs[1]x + coeffs[2] = 0

.. code:: python

    import torch
    from kornia.geometry.solvers import solve_quadratic

    coeffs = torch.tensor([[1., 4., 4.]])
    roots = solve_quadratic(coeffs)

.. note::
   In cases where a quadratic polynomial has only one real root, the output will be in the format [real_root, 0]. And for the
   complex roots should be represented as 0. This is done to maintain a consistent output shape for all cases.


.. autofunction:: solve_cubic

Finds the real roots of a cubic equation

.. math:: coeffs[0]x^3 + coeffs[1]x^2 + coeffs[2]x + coeffs[3] = 0

.. code:: python

    import torch
    from kornia.geometry.solvers import solve_cubic

    coeffs = torch.tensor([[32., 3., -11., -6.]])
    roots = solve_cubic(coeffs)

.. note::
   In cases where a cubic polynomial has only one or two real roots, the output for the non-real roots should be represented as 0. Thus, the output for a single real root should be in the format [real_root, 0, 0], and for two real roots, it should be [real_root_1, real_root_2, 0].
