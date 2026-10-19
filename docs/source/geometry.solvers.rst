kornia.geometry.solvers
=======================

.. meta::
   :description: The kornia.geometry.solvers module provides various solvers and optimizers for geometric problems. It includes polynomial solvers for quadratic and cubic equations, as well as functions for multiplying polynomials of different degrees and converting determinants into polynomial forms. These tools are essential for handling geometric transformations, optimizations, and other computational geometry tasks in computer vision.

.. currentmodule:: kornia.geometry.solvers

Affine, polynomial and homogeneous linear solvers used by the geometric estimators.

Affine estimation
-----------------

:func:`find_affine` fits a full 2D affine transform from at least three
non-collinear source points and their corresponding target points. It supports
weighted least-squares fitting and returns a source-to-target homogeneous
``(B, 3, 3)`` matrix. Use :class:`~kornia.geometry.ransac.RANSAC` with
``model_type="affine"`` when the correspondences contain outliers.

.. autofunction:: find_affine

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
gradient conventions. Under ``torch.compile`` with Inductor on CUDA, fused
multiply-adds can lose exact double roots; this known limitation is not planned
to be fixed, and the notes on :func:`solve_cubic` and :func:`solve_quartic` give
the rates and a workaround.

.. autofunction:: solve_quadratic
.. autofunction:: solve_cubic
.. autofunction:: solve_quartic
.. autofunction:: multiply_deg_one_poly
.. autofunction:: multiply_deg_two_one_poly
.. autofunction:: determinant_to_polynomial
