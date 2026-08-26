API Stability Policy
====================

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
SAM, and friends in ``kornia.feature``/``kornia.models``), ``kornia.onnx``,
and the multi-framework ``kornia.transpiler``. These track external
checkpoints, upstream repositories, and PyTorch internals; they are kept
working on currently supported PyTorch versions, but their interfaces and
weights may change with their upstreams, and they may be frozen or split out
rather than grown.

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

Importability is not publication
--------------------------------

Python exports whatever a module happens to bind, and ``import`` statements
bind names too. A file that does ``from torch import Tensor`` makes
``thatmodule.Tensor`` importable without anyone deciding that it should be.
Most of what you can import from a kornia submodule is of that kind: plumbing
a file needed, not API the project chose.

The rule above settles those cases -- undocumented means private -- and it is
the rule, not an aspiration. In 0.8.3 ``kornia.geometry.transform.pyramid``
stopped exposing ``pad``, which had never been anything but
``torch.nn.functional.pad`` leaking out of an import line. Downstream code
importing it broke. That name was never public API, kornia did not restore it,
and the fix belonged in the calling code.

``__all__`` states the same thing in a form tools can read. Where a module
defines it, it is that module's export declaration, and the project treats it
as binding in both directions:

* a name in ``__all__`` is covered by promise 1 below -- it gets a deprecation
  window before it is removed;
* a name that is not in ``__all__`` and not in the rendered documentation is
  private, and may disappear in any release without a deprecation window.

CI enforces the first half. ``.github/scripts/check_import_surface.py``
compares the importable surface of a pull request against its merge base and
fails the build when a name in ``__all__`` disappears. Removals of undeclared
names are reported in the same output but do not fail the build: they are not
API, and treating them as API would freeze thousands of incidental imports
forever. The report exists so a reviewer can still notice when a dropped name
looks load-bearing -- the judgement that was missing when ``pad`` went.

Names removed before this section was written
---------------------------------------------

Publishing this rule does not restore what earlier releases removed under it,
and kornia does not plan to re-add those names:

* ``kornia.core`` re-exported 32 torch names (``Tensor``, ``Module``, ``pad``,
  ``tensor``, ``zeros`` and friends) until 0.8.3. None was ever documented, so
  none was public by the definition above. Each has a direct torch replacement
  -- ``from kornia.core import Tensor`` becomes ``from torch import Tensor``,
  and ``pad`` becomes ``torch.nn.functional.pad`` -- and restoring them would
  rebuild the alias layer whose removal was the point.
* ``upscale_double`` was documented, and so *was* public. It was a leftover
  from the OpenCV-mimicking DoG implementation, superseded when that code was
  improved, and it was removed with no deprecation window. It stays removed;
  the role it served no longer exists.

Both removals predate this policy. What they should have had, and what every
future removal will have, is promise 3: a line in the release notes. Those
lines are being written retroactively.

If you depend on an undocumented name
--------------------------------------

Open an issue asking for it to be published. If it is a reasonable thing to
support it gets an ``__all__`` entry, a documentation entry and a test, and
from then on the promises on this page apply to it. That is a far better
outcome than finding out at upgrade time that the name was never ours to
keep, and it is cheap to ask -- please do that rather than pinning an old
release.

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
