API Stability Policy
====================

.. meta::
   :description: What Kornia users may rely on: stable, best-effort and experimental module tiers, the deprecation policy, and how breaking changes are announced.

Kornia is depended on by a large installed base (millions of downloads per
month) and, increasingly, by language models that carry a lagging snapshot of
the API in their weights. Both are punished by silent churn. This page states
what you may rely on, per module tier, and what change looks like when it has
to happen.

Stability tiers
---------------

**Stable core** — ``kornia.geometry``, ``kornia.augmentation``,
``kornia.color``, ``kornia.filters``, ``kornia.enhance``, ``kornia.losses``,
``kornia.metrics``, ``kornia.morphology``, ``kornia.io``, ``kornia.image``,
``kornia.core``, and the classical (non-pretrained) parts of
``kornia.feature`` (e.g. detectors, descriptors, matching utilities). The
promises below apply in full.

**Best-effort** — pre-trained model wrappers (LoFTR, LightGlue, DISK, DeDoDe,
SAM, and friends in ``kornia.feature``/``kornia.models``) and ``kornia.onnx``.
These track external checkpoints, upstream repositories, and PyTorch
internals; they are kept working on currently supported PyTorch versions, but
their interfaces and weights may change with their upstreams, and they may be
frozen or split out rather than grown.

**Experimental** — ``kornia.contrib`` and anything underscore-prefixed or
absent from the rendered documentation. No stability promise.

Planned evolution: support tiers
--------------------------------

The `roadmap <https://github.com/kornia/kornia/blob/main/ROADMAP.md>`_ describes
a future, finer-grained tier structure ("Tier A / B / C") in which a deliberately
small Tier A core carries per-symbol, CI-enforced guarantees. **That structure is
not in force yet.** It will be published only after the coordinated
repair-and-deprecation window described in the roadmap has resolved the
convention bugs surfaced by the ongoing per-operator audit — targeted within the
roadmap's ~6-month medium-term horizon (dates are intentions, not commitments).
Until the tier policy is published, this page is the authoritative stability
contract and the tiers above (stable core / best-effort / experimental) are what
you may rely on.

The repair window itself runs under the promises on this page, not around them:
a deliberate change to a default or convention that works as documented gets the
promise-2 deprecation treatment — at least one minor release of warnings, landing
at a 0.x minor boundary, with old-vs-new reference vectors. A fix for a
correctness bug (NaN, a crash, wrong math, or a result that contradicts what the
operation is documented to do) uses the escape hatch below instead: it lands when
it is ready, without a warning cycle, and its release note says what the old
behavior was.

What counts as public API
-------------------------

A symbol is public when it appears in the rendered documentation at
`kornia.readthedocs.io <https://kornia.readthedocs.io/en/latest/>`_.
Underscore-prefixed modules and names, and undocumented helpers, are private
regardless of importability.

The promises (stable core)
--------------------------

1. **Deprecation before removal.** A public symbol is never removed or given
   an incompatible signature in a single step. It first spends at least one
   minor release as a deprecated alias -- ``kornia.core._compat.deprecated``, or a shim
   with the old signature where the old function scripted with TorchScript -- which
   keeps the old call working while emitting a :class:`DeprecationWarning`
   that names the replacement and the version that introduced the
   deprecation.
2. **No silent semantic changes.** Changing a default that alters numerical
   output (interpolation flags, ``align_corners``, padding modes, coordinate
   conventions) counts as a breaking change and gets the same
   deprecation-window treatment. The historical ``align_corners``
   default flip, which changed results without erroring, is the exact
   pattern this rule forbids.
3. **Release notes carry the ledger.** Every deprecation, removal, and
   behavior change in the stable core is listed in the release notes of the
   release that introduces it.
4. **Imports stay clean.** Importing kornia emits no warnings of its own —
   none raised or triggered by kornia's import path, once PyTorch itself is
   loaded. This is enforced by CI-gated tests (``tests/test_import.py``), so
   downstream ``-W error`` users are safe.
5. **Conventions are stable.** The documented
   :doc:`conventions </get-started/conventions>` (tensor layout, coordinate
   order, angle units, homography conventions) are part of the API surface
   and covered by the same rules.

What the stable core promises for ONNX export and ``torch.compile``, and where
CI checks it, is stated on the :ref:`support page <export-support-promise>`.

.. _torchscript-support:

TorchScript
-----------

PyTorch has deprecated TorchScript in its documentation since 2.5, and from 2.15
``torch.jit.script`` and ``torch.jit.trace`` emit a ``FutureWarning``. Use
``torch.compile`` or ``torch.export`` for new code. Kornia keeps the TorchScript
support it has, and adds none:

- A function that compiles with ``torch.jit.script`` today keeps compiling, and
  kornia's tests script many of them.
- A function that does not compile is not made to compile. A defect that appears
  only in scripted calls is documented here and not fixed.

**Scripted CUDA calls and traced calls skip value checks.** Some functions check
their argument values with ``torch._assert_async`` when the checked tensor is on
CUDA; on the CPU the same check raises
:exc:`~kornia.core.exceptions.AsyncCheckError`. TorchScript removes that operator
when it compiles a function (checked with torch 2.5.1, the oldest supported
release, and 2.15), so a scripted call runs without the checks on CUDA tensors,
whether checks are enabled or not. A check on a CPU tensor still runs in a
scripted call, even with checks disabled, and raises ``torch.jit.Error``: :func:`~kornia.enhance.histogram` and
:func:`~kornia.enhance.histogram2d` check a CPU ``bandwidth`` on the CPU even when
the samples are on CUDA. Shape and type checks (``KORNIA_CHECK*``) still run in
scripted calls on every device. With torch 2.15, these scripted calls on CUDA take
an invalid argument without kornia's error:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Function
     - Invalid argument
     - Scripted call on CUDA tensors
   * - :func:`~kornia.enhance.adjust_gamma`
     - negative ``gamma`` tensor
     - returns a finite result
   * - :func:`~kornia.enhance.adjust_contrast`
     - negative ``factor`` tensor
     - returns a finite result
   * - :func:`~kornia.enhance.histogram`
     - ``bandwidth <= 0``
     - returns a result; it is finite for a negative ``bandwidth``, and NaN where a
       sample sits on a bin center and ``bandwidth`` is zero
   * - :func:`~kornia.enhance.histogram2d`
     - ``bandwidth <= 0``
     - as :func:`~kornia.enhance.histogram`
   * - :func:`~kornia.geometry.subpix.render_gaussian2d`
     - zero ``std``
     - returns non-finite values
   * - :func:`~kornia.enhance.equalize`, :func:`~kornia.enhance.equalize3d`
     - image values more than about 1/255 outside ``[0, 1]``, which the 256-level
       lookup cannot index
     - fails with PyTorch's out-of-bounds index device-side assert (raised in
       ``ScatterGatherKernel.cu`` with torch 2.15) instead of kornia's message; the
       CUDA context is unusable afterwards

The modules that wrap these functions, :class:`~kornia.enhance.AdjustGamma` and
:class:`~kornia.enhance.AdjustContrast`, behave the same when scripted.

A graph recorded with ``torch.jit.trace`` drops the operator too, on every device:
the checks run on the values traced, and not when the traced graph runs again. A
traced call with an invalid argument then gives the results in the table on the CPU
as well as on CUDA; on the CPU, ``equalize`` and ``equalize3d`` fail with PyTorch's
out-of-bounds index error instead of the device-side assert.

:func:`~kornia.enhance.solarize`, :func:`~kornia.enhance.posterize`,
:func:`~kornia.enhance.equalize_clahe` and :func:`~kornia.filters.gaussian_blur2d`
make the same kind of check but do not compile with ``torch.jit.script``.

To keep the checks, call the function eagerly with checks enabled (see
:func:`~kornia.core.check.disable_checks`), or through ``torch.compile`` with checks
enabled: Inductor keeps the assert from torch 2.9 on, and earlier releases compile it
to nothing. ``gaussian_blur2d`` skips its ``sigma`` check under ``torch.compile``.
Otherwise, validate the arguments before a scripted, traced or compiled call.

Versioning caveat
-----------------

Kornia is pre-1.0 and follows semver's 0.x semantics: minor releases (0.8 →
0.9) are where deprecation windows close and breaking changes land; patch
releases (0.9.0 → 0.9.1) are fixes only. The promises above are what makes
0.x livable: you get at least one minor release of warnings before anything
breaks.

Escape hatch
------------

A change that fixes a correctness bug (wrong math, wrong convention versus
the documented one) or a security issue may ship without a deprecation
window. When that happens the release notes say so explicitly.

Recording a completed deprecation
---------------------------------

Once the deprecation window has passed, remove the API and describe the removal
in a ``changelog.d/<PR>.breaking.md`` fragment. Update any existing entries in ``tests/api_surface.json``
in the same pull request; keep the module key, using an empty list if necessary.
The import-surface check verifies that recorded names leave the export surface
in that change. An inventory edit acknowledges an ``__all__`` removal only for
the exact same module and name; it does not authorize submodule removals.

Each module that declares ``__all__`` under a recorded package root has its own row.
To delete such a module, keep its key with an empty list: an empty row for a module
that no longer exists records the deletion, and stays in the inventory.

For an ``__all__`` name of a module that has no row of its own, such as a module
outside the recorded package roots, record the exact module and removed name in
``tests/api_surface_removals.json``. For example::

    {
      "kornia.feature.dedode": ["DeDoDe"]
    }

Only module/name pairs newly added relative to the pull request's merge base
can acknowledge removals, and each must match an actual ``__all__`` removal in
the same change. Existing entries cannot authorize a later removal. When reintroducing an API,
remove its obsolete acknowledgement so a future removal requires a fresh entry.
This file does not replace updates to the inventory or the deprecation and release-note
requirements above.
