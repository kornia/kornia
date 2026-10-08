kornia.geometry.solvers
=======================

.. meta::
   :description: The kornia.geometry.solvers module provides various solvers and optimizers for geometric problems. It includes polynomial solvers for quadratic and cubic equations, as well as functions for multiplying polynomials of different degrees and converting determinants into polynomial forms. These tools are essential for handling geometric transformations, optimizations, and other computational geometry tasks in computer vision.

.. currentmodule:: kornia.geometry.solvers

Polynomial and homogeneous linear solvers used by the geometric estimators.

Homogeneous Solvers
-------------------

.. autofunction:: null_vector_3x4

Polynomial Solvers
------------------

The polynomial solvers accept coefficients in descending degree order and return
real roots with multiplicity, using zero padding for missing real roots. The
public quadratic, cubic and quartic paths support fixed-shape graph capture,
including mixed batches with lower-degree rows. Near repeated roots, input
coefficient rounding can change whether a pair is real or complex; correctness
is defined by the represented coefficients. See each function's precision and
gradient conventions.

.. autofunction:: solve_quadratic
.. autofunction:: solve_cubic
.. autofunction:: solve_quartic
.. autofunction:: multiply_deg_one_poly
.. autofunction:: multiply_deg_two_one_poly
.. autofunction:: determinant_to_polynomial
