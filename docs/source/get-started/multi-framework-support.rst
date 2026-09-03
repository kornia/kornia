Multi-framework support
========================

.. meta::
   :description: Kornia no longer offers kornia.to_tensorflow(), kornia.to_jax() or kornia.to_numpy(). Testing found the Ivy-powered transpiler unreliable, so the feature and its docs were removed.

Removed: this page used to document ``kornia.to_tensorflow()``, ``kornia.to_jax()`` and
``kornia.to_numpy()`` — functions that lazily transpiled the whole library to another framework,
powered by the third-party `Ivy <https://github.com/ivy-llc/ivy>`_ project. In September 2026 we
tested that claim end to end, found it did not hold up, and removed the functions along with the
advertising in the README and on the landing page. This page now records what was found, for
anyone who lands here from an old link.

What we found
--------------

Testing used kornia 0.9.0rc1, ivy 1.0.0.5 (its latest release, from June 2025), torch 2.14.0,
tensorflow 2.21.0, and jax 0.10.2 / jaxlib 0.10.2 — following the exact examples this page used to
show.

* ``kornia.to_tensorflow()``, ``to_jax()`` and ``to_numpy()`` crashed immediately with
  ``TypeError: unhashable type`` when kornia was checked out into a directory whose path contains
  the substring ``kornia`` — the default name ``git clone`` gives the repository. Ivy's transpiler
  decides whether a module belongs to kornia with a plain substring check against each module's
  file path, so it also recursed into unrelated parts of PyTorch's dependency tree (``ctypes``,
  ``dill``, ``unittest.mock``, …) and hit an object it could not hash.
* Even from a differently named checkout, ``to_tensorflow()`` segfaulted during transpilation
  whenever ``transformers`` was installed in the same environment — which it was by default for
  kornia contributors, since both packages shipped in the same ``dev`` extra. The crash happened
  inside Ivy's Hugging Face integration lookup, which imports ``triton`` as a side effect.
* With ``transformers`` absent and the environment's ``ruff`` executable on ``PATH`` (Ivy shells
  out to it to format generated code), ``to_tensorflow()`` did transpile and run
  ``rgb_to_grayscale`` correctly.
* ``to_numpy()`` transpiled but failed at call time: the generated code checked
  ``np.bfloat16``, an attribute NumPy does not have.
* ``to_jax()`` failed outright: Ivy requires ``flax>=0.8.0``, which this page never mentioned;
  after installing it, the call failed instead with
  ``module 'jaxlib' has no attribute 'xla_extension'``, an incompatibility between Ivy and current
  ``jaxlib`` releases.

None of these were bugs in kornia's own code — all five are in Ivy's transpiler or in its
compatibility with current TensorFlow/JAX/NumPy releases.

Is Ivy still maintained?
--------------------------

As of September 2026, `ivy-llc/ivy <https://github.com/ivy-llc/ivy>`_ (which now redirects to
`unifyai/ivy <https://github.com/unifyai/ivy>`_) had a single commit in the previous twelve
weeks — a rebrand, not a fix — and its last PyPI release, ``1.0.0.5``, shipped in June 2025. It
carries close to a thousand open issues, including unmerged fixes for exactly the kind of
NumPy/JAX version drift this page ran into. The `unifyai <https://github.com/unifyai>`_
organization itself is active, but its recent development effort is going into an unrelated
product line, not Ivy.

If Ivy becomes reliable again, multi-framework support is worth revisiting — but as a fresh
integration, not a restoration of ``kornia.to_tensorflow()``/``to_jax()``/``to_numpy()`` from this
page. Their implementation is gone; ``kornia.to_tensorflow()``, ``kornia.to_jax()``,
``kornia.to_numpy()`` and their ``kornia.transpiler`` counterparts now only raise a clear error
pointing here if something still calls them.
