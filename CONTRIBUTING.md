# Contributing to kornia

## Our social contract

We (maintainers) develop kornia for ourselves and the computer vision community.
If kornia is useful to you in any way — to import, to prototype, to use as a reference for your agent — that makes us happy.
You are welcome to help us improve kornia: report a bug, implement a feature, propose an idea, answer a question, or simply tell others about the project.
We try to do our best to accept your help — by reviewing PRs, fixing bugs, and so on — but we do not promise to review every PR, let alone quickly.
We have jobs, families, health problems, and holidays, just like everyone else.

You may use AI to do the work — so do we. There are no approved or disapproved amounts of AI use. Tell us how AI helped; we do the same.
Above all, we ask you to care. Good work done with care makes the world better; sloppy work makes the world worse.

The rest of this guide contains practical and technical guidance so nobody has to guess. If you have experience with open source, you probably already know most of it.

Coding agents should also follow the [execution checklist in AGENTS.md](AGENTS.md).

## Practical guidelines

- Search existing issues and pull requests before starting. Focused fixes and documentation improvements are welcome as pull requests. For a new public API, a behavior change, or a large piece of work, an early issue or discussion can save everyone time. A conversation helps us align; it is not a promise that somebody will review or merge the result.
- You do not need to ask before working on an issue that is not reserved. Say on the issue that you are taking it, then open a pull request, as a draft if you still have questions. If a claim has no pull request after two weeks, anyone may take the issue; say so on the issue when you do.
- Some issues say they are reserved for a maintainer implementation, and some already have an open pull request. Please comment on the issue or on that pull request instead of opening another one.
- Follow [Correctness and backend support](ROADMAP.md#correctness-and-backend-support) before implementing a numerical fix or backend workaround. Repairs within the supported contract are welcome directly as pull requests, subject to existing issue reservations. Do not add special paths solely for synthetic extremes: half-precision overflow far beyond image/coordinate ranges, `1e20` float32 image values or pixel coordinates, or a matrix scaled by `1e15`. These are examples, not universal magnitude cutoffs. First obtain an explicit maintainer scope decision on an issue with a concrete application and reproducer. A failing synthetic test alone does not justify expanding support.
- Kornia functions take floating-point tensors, and images are floats in `[0, 1]` (see [Conventions & Pitfalls](https://kornia.readthedocs.io/en/latest/get-started/conventions.html)). Integer input is not a goal: a few cheap, scale-free image filters accept it, and we do not accept pull requests that add integer-input support anywhere else, especially for `int8` and other narrow integer types. If a function misbehaves on an integer tensor, convert the input to float first.
- Check the part you changed and tell us what you ran. A bug fix is easier to trust when it includes a test that fails without the fix. Numerical work is easier to review when expected values come from a reference implementation, a paper, or a mathematical invariant.
- Look for an existing `kornia` utility before adding another implementation. Keep changes focused, follow nearby code, and update the documentation when public behavior changes.
- Read your own contribution and stay part of the conversation about it. You do not need perfect English or an immediate answer; we care that questions and technical concerns are treated seriously.
- If you are contributing as part of an application, remember that we read the work and review conversation, not pull request counts.
- Keep project decisions and review discussions in public GitHub issues, discussions, or pull requests so others can find and learn from them.
- Review and closure decisions belong to maintainers; see [Review Process](#review-process).

## Ways to Contribute

- Ask or answer questions in [GitHub Discussions](https://github.com/kornia/kornia/discussions),
  [PyTorch Discuss](https://discuss.pytorch.org) (`#kornia`), or [Discord](https://discord.gg/HfnywwpBnD).
  Use issues for bug reports rather than Q&A.
- Report bugs through the [issue templates](https://github.com/kornia/kornia/issues/new/choose),
  with a minimal reproduction and the requested environment details. Follow the
  [privacy-conscious diagnostic guidance](TESTING.md#diagnosing-reported-problems).
- Find work in the [roadmap](ROADMAP.md#areas-seeking-contributors) and
  [help wanted issues](https://github.com/kornia/kornia/labels/help%20wanted); check reservations first.
- Support the project through [Open Collective](https://opencollective.com/kornia),
  [GitHub Sponsors](https://github.com/sponsors/kornia), or CUDA testing resources.

## Developing kornia

### Setup

Fork [kornia](https://github.com/kornia/kornia/fork), clone your fork, and branch from upstream:

```bash
git clone git@github.com:<your GitHub username>/kornia.git
cd kornia
git remote add upstream https://github.com/kornia/kornia.git
git fetch upstream
git switch -c fix/my-fix upstream/main
```

Install [Pixi](https://pixi.sh), then set up the project environment:

```bash
pixi install
pixi run -e default install
# Optional versioned environments
pixi install -e py312
pixi run -e py312 install
pixi install -e py313
pixi run -e py313 install
# CUDA development
pixi run -e cuda install
```

Python dependencies under test live in `.venv` or `.venv-py*`, not `.pixi`.
Before CUDA work, read the [environment and CUDA installation details](TESTING.md#environment-and-validation-tasks),
including resynchronization, Windows wheel replacement, and availability checks.

### Development Workflow

Keep each PR focused. Search for existing Kornia operations, follow nearby contracts,
and run focused validation before broad checks:

```bash
pixi run test-module tests/path/to/test_file.py --device=cpu --dtype=float32,float64
pixi run pre-commit-all
pixi run typecheck
```

Run full pre-commit checks before presenting a code change as finished; `pixi run lint`
checks only Ruff. [TESTING.md](TESTING.md) owns test selection, device/dtype profiles,
compile/ONNX coverage, doctests, and revision comparisons. Performance changes use the
same benchmark on base and branch under [the benchmark methodology](benchmarks/README.md).

## Contributing to Documentation

Edit `docs/`, build with `pixi run -e default build-docs`, and preview
`docs/build/html/index.html`. Read the Docs treats Sphinx warnings as failures and uses
the locked docs dependencies. Public classes/functions must appear in the corresponding
`docs/source/*.rst` API page. For a new feature detector or descriptor, also update the
`responses` list and matching visualization branch in `docs/generate_examples.py`,
following DISK/ALIKED/XFeat and preserving `(B, 3, H, W)` images.

### Docstring style

A docstring is how a reader, human or agent, decides which function to call, so it states the choice before the details. `kornia/geometry/depth.py` is the worked example. Docstrings are reST: write literals with double backticks, as in ``` ``(B, 1, H, W)`` ```, and keep single backticks for roles such as `` :func:`name` ``.

- **Title line**: one physical line, in task words, imperative, naming what sets the function apart from its siblings (2D or 3D, channels first or last, the criterion, one call or module form). Aim for about 80 characters; 120 is the hard limit. Avoid markup that breaks reST, such as a bare `*`. No two unrelated public symbols may share one (`tests/test_docstring_first_lines.py`); a class and the function it wraps may.
- **Status words** come first when they apply: `Deprecated: use X.` or `Alias of X.`. An alias whose whole job is to forward to another object has the title line ``Alias of :func:`target`.``, and a deprecated object the title line ``Deprecated: use :func:`target`.``, with ``:class:`` or another Python-domain role for a target that is not a function; either title needs no `See Also:`. The target is another public kornia object, `Deprecated: use` belongs to objects that `@deprecated` wraps, and `Alias of` to an undecorated function whose body only returns the target called with its own arguments, unchanged, or to a class that subclasses the target and adds nothing. Do not label a function "Legacy"; for a nearby better choice, write a neutral "for X, use Y" in the contract paragraph.
- **Contract paragraph** (2-5 sentences) starts from the call: ``` ``points = depth_to_3d(depth, camera_matrix)`` ``` takes ... and returns .... A callable module shows construction and invocation; a data class (`Boxes`, `PinholeCamera`) shows construction and its main method. Name inputs and output with shapes, frames and units, and end with "use X instead" when a nearby task is better served elsewhere.
- **Existing blocks** stay: `.. note::`, `Convention:`, `Args:`, `Return:`/`Returns:`, `Raises:`, trimmed where the paragraph already says the same. In a long class docstring, put whatever decides correct use in the first paragraph.
- **`See Also:`** goes after `Returns`/`Raises` and before `Example`.
  - Write each entry as `` - :func:`name`: how it differs. ``; the explanation is required.
  - Spell targets as Sphinx registers them: a short name for an object documented under the same module, the documented path (`~kornia.geometry.camera.perspective.unproject_points`) for the rest. A shorter import alias renders as plain text, with no warning.
  - List kornia objects only, with no self-links or private names; mention torch and Python functions in the contract paragraph instead. Link true siblings both ways (v1 and v2, 2D and 3D); reviewers check that, no test does.
- **Module docstrings** open with a title line, then the conventions the module's functions share, then "Which function to use", grouped by task, one line each, naming every entry of the `__all__` of the namespace users import from as a role (`` :func:`name` ``) or a literal (``` ``name`` ```). A namespace without `__all__` (`kornia.geometry`, `kornia.geometry.transform`) catalogs its subpackages and main entry points instead; a defining file without `__all__` gets no catalog.
- **Shapes** are literals such as ``` ``(B, 1, H, W)`` ```. Keep `:math:` for formulas, and for a tuple that is a value, such as a point's `` :math:`(X, Y, Z)` ``; write a value range as an interval, `` :math:`[0, 1]` ``.

Enforcement, rollout tables, and reference-resolution rules are in [Docstring checks](TESTING.md#docstring-checks).

## Coding Standards

- Use existing Kornia operations and nearby batching, broadcasting, shape, dtype, and device conventions.
  Preserve the caller's dtype/device; do not create implicit CPU/default-dtype tensors.
- Public APIs need annotated inputs/outputs, docstrings, exports, and rendered API entries.
  Use `torch.Tensor` and `torch.nn.Module` directly where practical; follow TorchScript's
  stricter types in scripted modules. Non-JIT modules use `from __future__ import annotations`.
- Keep 120-character lines, Apache 2.0 source headers, and English comments that explain why.
  Ruff and `ty` enforce style and types. Prefer clear code; profile before optimizing.
  Validate autograd, aliasing, and compile behavior before introducing in-place operations.
- For new algorithms, name a defining paper or reference implementation. Do not copy a
  reference's bug: decide the intended behavior and document deliberate departures.
  For example, Kornia's non-maxima suppression keeps strict maxima.
- Avoid new runtime dependencies. Explain why the existing stack is insufficient and
  consider install, build, and platform costs before proposing one.
- Follow [the stability policy](https://kornia.readthedocs.io/en/latest/get-started/stability.html)
  for API changes and [Public API inventory](TESTING.md#public-api-inventory) for export removals.
  Inventory acknowledgements do not replace deprecation or release notes.
- Follow [Test Structure](TESTING.md#test-structure) and the relevant device/precision guidance
  for tests, and [Dynamic-shape check messages](TESTING.md#dynamic-shape-check-messages) when adding validation.

### Changelog entries

User-visible changes need a fragment; internal changes may explain their omission and
use `no-changelog`. Follow [the fragment guide](changelog.d/README.md) for names, types,
pre-PR filenames, and release handling. Only release PRs assemble `CHANGELOG.md`.

### Closing a documented defect

Before merging a fix that closes an issue, search its number in references and test-name
suffixes (replace `NNNN` with the issue number):

```bash
grep -rnE '(#|issues/)NNNN([^0-9]|$)|test_[[:alnum:]_]*_NNNN([^0-9]|$)' kornia/ tests/
```

This includes names such as `test_wart_example_NNNN` without matching bare numerical constants.
Check every hit: stale defect prose
in docstrings or errors must be corrected, strict xfails removed when fixed, and wart
assertions updated. Tests may keep the issue number to record the regression. Search
by number, not wording; a link label and its URL can disagree. No test hits is a valid
result, not a reason to invent coverage. See the test-name conventions in nearby files.

## Pull Request

Fill in the [PR template](.github/pull_request_template.md) in the PR body, never a committed
`PR.md`. Describe the behavior, evidence, compatibility/performance tradeoffs, and checks
run or unverified. Include a failing-before regression when appropriate and explain the
independent expected result for numerical work. Keep generated logs, benchmark output,
and investigation notes outside the library diff.

### Working with AI

AI tools are welcome. Check their output as you would your own and describe the actual
human contribution and its effect: steering, diagnosis, design decisions, edits, review,
or supervision. Arranging another model's review is distinct from personally verifying
the result; credit AI-generated findings and tests to the AI. Merely requesting the task
or directing an agent to follow reviews can mean zero substantive human technical
contribution. The PR template asks for this distinction, not a percentage or a chat transcript.

Every agent-authored GitHub PR description, issue, comment, and review on a user's behalf
ends with `Posted on behalf of @<github-handle> by <Agent> (<actual model>).` Use the actual
handle, agent, and model rather than a generic generated-by footer.

### Review Process

Review your diff and validation before submission. Maintainers review PRs; contributors
should report problems in another contributor's PR on its linked issue rather than post
reviews or approvals. Respond to feedback and explain decisions; maintainers may request
a smaller scope or decline work, and review or merge is not guaranteed.

PRs are marked stale after 15 quiet days as a triage signal; activity clears the label,
and maintainers can use `pinned` to exempt a PR. There is **no automatic closure for
inactivity**. A maintainer may close a quiet PR at their discretion; it may be reopened.
A quiet PR may be waiting on us, so the label is a nudge to follow up, not a deadline.

Code changes run CPU, compile, and type checks; pre-commit checks formatting, spelling,
headers, and file hygiene. Documentation changes run the docs build. Every non-draft PR
gets automated Copilot review when opened or marked ready and needs a maintainer approval.
PRs are squash-merged. Fix failures caused by your change or explain them.

## License

By contributing, you agree to license your contributions under the Apache License. See [LICENSE](./LICENSE).
