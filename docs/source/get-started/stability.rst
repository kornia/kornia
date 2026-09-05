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
every semantic or default change it ships gets the promise-2 deprecation
treatment — at least one minor release of warnings, landing at a 0.x minor
boundary, with old-vs-new reference vectors — and only clearly-broken-output
bugs (NaN, crashes) use the correctness escape hatch below.

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
   minor release wrapped in ``kornia.core._compat.deprecated``, which
   keeps the old call working while emitting a :class:`DeprecationWarning`
   that names the replacement and the version that introduced the
   deprecation (this is the mechanism the 0.8.3 ``kornia.utils`` →
   ``kornia.image`` migration used).
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
in ``CHANGELOG.md``. Update any existing entries in ``tests/api_surface.json``
in the same pull request; keep the module key, using an empty list if necessary.
The import-surface check verifies that recorded names leave the export surface
in that change. An inventory edit acknowledges an ``__all__`` removal only for
the exact same module and name; it does not authorize submodule removals.

For a submodule API recorded only under an ancestor package, or an API absent
from the inventory, record the exact module and removed ``__all__`` name in
``tests/api_surface_removals.json``. For example::

    {
      "kornia.geometry.boxes": ["Boxes"]
    }

Only module/name pairs newly added relative to the pull request's merge base
can acknowledge removals, and each must match an actual ``__all__`` removal in
the same change. Existing entries cannot authorize a later removal. When reintroducing an API,
remove its obsolete acknowledgement so a future removal requires a fresh entry.
This file does not replace updates to the inventory or the deprecation and release-note
requirements above.
