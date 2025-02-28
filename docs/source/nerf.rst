.. meta::
   :name: description
   :content: "The Kornia.nerf module provides tools for working with Neural Radiance Fields (NeRF), including models, solvers, renderers, and samplers. Key components include NerfModel, NerfSolver, and VolumeRenderer for high-quality 3D scene rendering."
kornia.nerf
===========

The functions in this sections perform Neural Radiance Fields (NeRF) related

.. currentmodule:: kornia.nerf

Models
------

.. autoclass:: kornia.nerf.nerf_model.NerfModel
    :members:
.. autoclass:: kornia.nerf.nerf_model.NerfModelRenderer
    :members:
.. autoclass:: kornia.nerf.nerf_model.MLP
    :members:

Solvers
-------

.. autoclass:: kornia.nerf.nerf_solver.NerfSolver
    :members:
    :exclude-members: init_training

Renderers
---------

.. autoclass:: kornia.nerf.volume_renderer.VolumeRenderer
    :members:
.. autoclass:: kornia.nerf.volume_renderer.IrregularRenderer
    :members:
.. autoclass:: kornia.nerf.volume_renderer.RegularRenderer
    :members:

Samplers
--------

.. autoclass:: kornia.nerf.samplers.RaySampler
    :members:
.. autoclass:: kornia.nerf.samplers.RandomRaySampler
    :members:
.. autoclass:: kornia.nerf.samplers.RandomGridRaySampler
    :members:
.. autoclass:: kornia.nerf.samplers.UniformRaySampler
    :members:
