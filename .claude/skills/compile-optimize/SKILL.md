---
name: compile-optimize
description: Use when making a kornia op torch.compile / dynamo compatible, fixing a graph break, or optimizing a function for speed. Codifies the compile-first workflow — genuine fullgraph fixes (no is_compiling hacks), fullgraph==eager verification, and a required before/after benchmark.
---

# Compile-first optimization (kornia)

Make a kornia function genuinely `torch.compile(fullgraph=True)`-compatible and measurably faster. This is a **rigid** workflow — follow every step.

## The non-negotiable rule: no `is_compiling()` hacks

`torch.compiler.is_compiling()` guards are **banned** as the fix. They make the compiled path *skip* the data-dependent code instead of making it traceable, so eager and compile run different paths and checks are silently dropped under compile. That is "fullgraph by exclusion," not genuine fullgraph — and it leaves the op only partially compiled, killing the speedup.

The fix must run the **same code path** in eager and compile.

## Workflow (create a TodoWrite item per step)

1. **Reproduce the break.** `torch.compile(fn, fullgraph=True)(x)` and capture the exact graph-break location (a traceback frame in `kornia/…`). Never guess.

2. **Classify the break and apply the genuine fix:**

   | Break | Genuine fix (single path) |
   |---|---|
   | `if (t < 0).any(): raise` — value **validation** | `torch._assert_async((t >= 0).all(), "msg")` — keeps the check, no break |
   | `bits==0`/`==8` style **logic branch** on a tensor | branchless `torch.where(cond, a, b)` — compute both, select |
   | `if not mask.any(): break` — **early-exit** in a fixed-bound loop | delete it; trailing iterations must be provable no-ops |
   | `if mask.all(): return None` — **early-out** returning a different type | always return the value; make the consumer's use a no-op |
   | `if batch_prob.sum()==1` — **gate** on a {0,1} tensor | branchless multiply (`gate * value`) |
   | Python `if p == 1` where `p` is a **Python** float/bool | leave it — resolved at trace time, not a break |

   If none apply (unbounded `while`, `.item()`-driven shapes, random-permutation dispatch, dynamic-shape `nonzero`/`unique`), it needs a **redesign or a maintainer decision** — do not force a hack. Document it and stop.

3. **Verify genuine fullgraph == eager.** Same path, matching output:
   ```python
   c = torch.compile(fn, fullgraph=True); torch.allclose(c(x), fn(x), atol=1e-5)
   ```
   For branchless rewrites of edge-cased logic, verify **exhaustively** across the edge inputs (e.g. every `bits` 0..8), not one sample.

4. **Benchmark before/after — REQUIRED for every touched function.** A compile fix that doesn't speed anything up (or regresses eager) must be justified.
   ```python
   import torch.utils.benchmark as bench
   def us(f,*a): return bench.Timer(stmt="f(*a)",globals={"f":f,"a":a}).blocked_autorange(min_run_time=1.0).median*1e6
   eager = us(fn, *args)
   c = torch.compile(fn, fullgraph=True); c(*args)  # warmup
   comp  = us(c, *args)
   ```
   Record eager µs, compiled µs, speedup at a realistic size (e.g. `(8,3,128,128)`). Also confirm the rewrite didn't **regress eager** (branchless "compute both branches" and deleted early-exits can add eager work — measure it). Put the numbers in the PR.

5. **Lock it in.** Add/confirm a `test_dynamo` for the op (pattern: build op → `op_optimized = torch_optimizer(op)` → `assert_close`). Note: dynamo tests only run when `KORNIA_TEST_OPTIMIZER=inductor` is set — normal PR CI does not run them, so verify locally.

6. **Verify the suite is unchanged.** Run the op's full test file with `--dtype=float32`. For shared helpers or the augmentation base, run the whole module suite — a branchless rewrite can shift RNG *consumption* (e.g. always drawing a sample that used to be conditional); the suite passing unchanged is the proof no seeded test depends on it.

7. **PR.** One op (or one shared helper) per PR. Body: the break, the genuine fix, the fullgraph==eager evidence, and the **benchmark table**. `ruff check` + `ruff format --check` clean (pinned version from `.pre-commit-config.yaml`).

## Leverage: prefer shared fixes

The biggest wins are shared helpers/base classes, where one genuine fix unblocks many ops:
- `kornia/augmentation/base.py` `__batch_prob_generator__` gate → unblocked ~13 augmentations.
- `kornia/losses/_utils.py` `mask_ignore_pixels` → unblocked dice/focal/tversky.

When an op breaks, trace to the deepest `kornia/` frame first — the break is often in a shared utility, and fixing it there is far higher leverage than per-op.

## What's already compile-clean

Most of the numeric core already compiles (filters, color, geometry transforms, losses, morphology, metrics). Sweep before assuming a break exists.
