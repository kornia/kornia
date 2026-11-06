# Agent guide for kornia

Use this file for execution. [CONTRIBUTING.md](CONTRIBUTING.md) owns contribution and
coding rules; [ROADMAP.md](ROADMAP.md) owns priorities and correctness policy;
[TESTING.md](TESTING.md) owns validation procedures and backend details.

## Start here

- Read the relevant issue decision and nearby implementation/tests before editing.
  Preserve existing user changes and keep the task focused.
- Library code lives in `kornia/`; reusable test support lives in `testing/`, not `tests/`.
  Check import cycles when changing exports: `kornia/__init__.py` loads `filters` and
  `geometry` before modules that depend on them.
- Search for an existing Kornia operation. Follow nearby shape, batching, dtype/device,
  and TorchScript conventions. Use the [coding standards](CONTRIBUTING.md#coding-standards)
  and [docstring style](CONTRIBUTING.md#docstring-style); public API changes need exports and docs.

## Environment and commands

Use Pixi. [Setup](CONTRIBUTING.md#developing-kornia) selects `.venv` or `.venv-py*` for
Python dependencies; `.pixi` supplies tools, not the packages under test. Read the
[CUDA installation notes](TESTING.md#environment-and-validation-tasks) before CUDA work.

```bash
pixi run test-module tests/path/to/test_file.py --device=cpu --dtype=float32,float64
KORNIA_TEST_OPTIMIZER=inductor pixi run test-module tests/path/to/test_file.py
pixi run pre-commit-all
pixi run typecheck
pixi run doctest
pixi run -e default build-docs
```

`test-module` forwards pytest node IDs and `-k`. Start focused, then cover affected
callers. Run full pre-commit before presenting a code change as finished; `lint` is
only Ruff. Report what passed and what remains unverified.

### Comparing a branch against another revision

Run comparison snippets from each worktree root as stdin or `-c`, using an explicit
interpreter. Print and verify `kornia.__file__` and `sys.executable` inside each probe:
an external script can silently import the primary editable checkout. Follow the
[comparison procedure](TESTING.md#comparing-a-branch-against-another-revision).

## Read when the change touches

| Area | Required guidance/action |
| --- | --- |
| Numerical or gradient tests | Use `BaseTester`, injected device/dtype, `self.assert_close()`, and `self.gradcheck()`; use `torch_optimizer` for compile coverage. See [Test Structure](TESTING.md#test-structure). Use independent hardcoded expected literals plus their generation snippet/source, not a runtime optional reference dependency. |
| Singular gradient guards | `clamp` alone is not portable; differentiate safe arguments and test the older supported Torch leg. See [gradient guards](TESTING.md#gradient-guards-across-pytorch-versions), including the float16-underflow case. |
| CPU/CUDA half precision | Follow the [half-precision profiles](TESTING.md#7-half-precision-dtypes-float16--bfloat16); CUDA needs per-test subprocess isolation. Fix/remove or rename manifest entries with the corresponding test changes. |
| MPS or TF32 | Read [MPS limitations](TESTING.md#8-mps-apple-silicon-limitations) and [TF32](TESTING.md#1-tf32-tensorfloat-32-precision) before adjusting skips/tolerances. The MPS manifest is a full-suite contract, not a focused-run baseline. |
| Checks or dynamic shapes | Avoid specializing sizes in error strings; follow [Dynamic-shape check messages](TESTING.md#dynamic-shape-check-messages), including its TorchScript exception. |
| Public exports | Follow [Public API inventory](TESTING.md#public-api-inventory); exact removal acknowledgements do not replace deprecation or release notes. |
| Compile or ONNX | Run the [compile](TESTING.md#torchcompile-conformance-5819) and [ONNX](TESTING.md#onnx-conformance-5819) promised-package selections for touched packages and their importers. Inspect the ONNX subsystem/mixins before export changes and verify runtime as well as graph construction. |
| Performance | Follow [benchmarks/README.md](benchmarks/README.md), use public APIs and shared helpers, and benchmark base and branch on the same workloads. |

## Correctness triage and fixes

Apply [Correctness and backend support](ROADMAP.md#correctness-and-backend-support) in order:

1. Check the contract and equivalent plain-PyTorch operation where applicable. Classify
   the report: math/semantics, numerical formulation, upstream behavior, or no defect.
2. Check scope and existing solutions before coding. Stop for excluded inputs until an
   explicit scope decision; demonstrate benefit over existing options before adding a solver.
3. For upstream failures, test latest stable and available next PyTorch on the affected
   backend; record gaps. Follow the roadmap's fixed-upstream rule before adding a workaround.
4. Name the workflow, inputs, dtype/device, and grad/autocast mode. Separate synthetic
   proof, real-data exposure, and downstream impact; do not invent user frequency.
5. Pin independent expected results in focused tests (`test_convention_*` for conventions).
   Use `self.assert_close()`'s dtype tolerances or justify an override. Report worse
   downstream results and unresolved convention choices for maintainer judgment.
6. Record branch rationale and workaround scope, retirement, and a short measured-cost
   summary beside the code; put detailed results and tradeoffs in the PR and accepted
   behavior changes in release notes. State
   the outcome explicitly, including a supported alternative or no change.

## Review and completion

- Treat a reviewer's suggested fix as a hypothesis. Check the existing invariant first:
  `Boxes._valid`, for example, already follows indexed reads/writes and zeros invalid
  exports; adding a separate count can create another synchronization problem.
- If two review rounds in a row find regressions introduced by the preceding fix,
  stop patching cases. State the invariant and redesign the responsible code before
  pushing again, reverting previous fixes when simpler.
- Follow the [PR and AI disclosure rules](CONTRIBUTING.md#pull-request), including the
  accurate human-work statement. End agent-authored GitHub posts with the literal form:
  `Posted on behalf of @<github-handle> by <Agent> (<actual model>).`
- Apply the [changelog rules](changelog.d/README.md), register public APIs in the matching
  docs page, and update detector/descriptor examples as described in
  [Contributing to Documentation](CONTRIBUTING.md#contributing-to-documentation).
- Before closing an issue, search its number through code/tests and repair stale defect
  prose and wart assertions; follow [Closing a documented defect](CONTRIBUTING.md#closing-a-documented-defect).
- Keep logs, measurements, plots, scratch data, and task reports outside the library diff.
  Inspect the complete PR file list before committing or pushing.

## Helping users diagnose problems

Start with a minimal reproduction and the narrowest test. Compare with the relevant
base before claiming a failure is pre-existing. Request only the relevant version,
OS, and device details from [Diagnosing reported problems](TESTING.md#diagnosing-reported-problems).
Ask users to preview/redact broader output; do not request private paths, usernames,
environment variables, tokens, internal hostnames, or private data/model names.
