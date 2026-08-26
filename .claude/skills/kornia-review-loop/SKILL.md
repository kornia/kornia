---
name: kornia-review-loop
description: Use when responding to code-review findings on a kornia PR — triaging each finding, deciding what is in scope, and pushing fix commits. Stops the fix commits from becoming the next round's findings (on kornia#4006, 20 of 24 post-round-2 findings were introduced by earlier fixes).
---

# kornia-review-loop: a fix commit carries the whole verification bar

Every review round on a kornia PR is answered by ONE push that has been verified to the same
standard as the original PR. This is a **rigid** workflow — create a todo per step and do them in
order. The reviewer's model does not matter; what matters is that the code you push has been
attacked before they see it.

## Why this exists

On kornia#4006 the reviews were accurate. The fixes were the problem: the same half-precision
rounding bug was "fixed" three times one layer deeper, an empty-path guard flipped from too strict
to too lax to device-specific over four rounds, and two regression tests passed vacuously. On
kornia#4028 (docs-only) every finding after round 2 was against a rule added mid-PR. Both PRs
would have converged in two rounds if each push had been gated.

## Workflow

1. **Record the pre-fix SHA.** `git tag -f prefix-<round> HEAD` before touching anything, where
   `<round>` is this review round's number (1, 2, ...). A local tag, not a `/tmp` file, so it
   survives a shell restart without leaving scratch state in the kornia tree. Never push it —
   `git push --tags` or `--follow-tags` would leak it to the remote — and delete it
   (`git tag -d prefix-<round>`) once the round is answered.

2. **Triage every finding before writing code.** For each one, reproduce it on the branch AND on
   a worktree of `origin/main` (`git worktree add ../main-probe origin/main`). Three outcomes:
   - *regression* (fails on branch, passes on main) → fix it;
   - *pre-existing* (fails on both) → reply with both measurements, file or link an issue, do
     NOT fix it in this PR;
   - *wrong* → refute in writing with the measurement.
   A finding is never dropped silently; the next reviewer re-raises anything unanswered. Findings
   that arrive with confident numbers are still re-measured — one on #4006 attributed to the branch
   a divergence that was identical on main.

3. **Fix defects only.** If a finding makes you want to *add* something — a helper, a rule in
   `AGENTS.md`, a deprecation, a new test file for an adjacent function — write it as a follow-up
   issue and link it in the reply. Mid-PR additions are reviewed as fixes and generate rounds. The
   PR only shrinks or corrects during review.

4. **Write tests that cannot pass vacuously.** Use the `kornia-precision-testing` skill for any
   test touching dtype, device, capture or degenerate sizes. Before trusting a new regression test,
   check it FAILS on the pre-fix SHA — `git stash; pytest <test>; git stash pop` is a no-op once
   the fix is committed (which step 5 presumes): there is nothing to stash, the test runs on the
   already-fixed tree, passes, and you wrongly conclude it is vacuous. Use the step-1 tag instead:
   `git worktree add ../prefix-probe prefix-<round>`, copy the new test file into the probe
   worktree (it does not exist there at that tag), then run
   `(cd ../prefix-probe && PYTHONPATH=$PWD python -m pytest <test>)` and confirm it FAILS, then
   `git worktree remove ../prefix-probe`.

5. **Self-review the delta with a fresh agent.** Dispatch a subagent on
   `git diff prefix-<round>..HEAD` with this instruction: "Attack only the new code. For
   every changed line check: is a size cast into a half dtype before a division; does a
   degenerate/empty path validate less or more than the full path; is a `dtype=None` resolved
   before promotion is decided; does a new compile test carry `dynamo`/`compile` in its name; do
   MPS/half skips match the nearest existing test; did a docstring, comment, or changelog sentence
   next to the change stop being true; does the fix in one function need mirroring in its 3d/other
   sibling." Fix what it finds. Repeat once if it found anything.

6. **Gate: `pixi run verify-delta`.** Zero `new` failures on every available surface — the four
   `--only` names are `cpu float32`, `cpu float16,bfloat16,float64`, `mps float32`, and
   `inductor cpu float32` (runs `-k "dynamo or compile"`). Half precision and MPS have no CI job —
   this is their only signal. With no flags it derives test dirs from the diff, and a change under
   `testing/`, `conftest.py`, `pyproject.toml`, or `pixi.toml` widens to the whole suite (~20 min
   per surface per tree); scope it while iterating with
   `pixi run verify-delta -- --tests tests/geometry --only "cpu float32"`. Exit 0 = no new
   failures, 1 = new failures (`NEW <id>` lines name them), 2 = nothing was verified (no surface
   ran pytest on the branch side — a typo'd `--tests` path or an unavailable `--only` surface),
   which is a failure, not a pass. A diff that maps to no test directory at all (a docs-only PR)
   also exits 0 and prints "nothing to verify" — that is not a green gate; state in the reply which
   surfaces you instead ran by hand. A `N*` row means the path had no baseline on `origin/main` (a
   new test package), so every failure there counts as new; `skipped` means that surface was
   deselected or unavailable. Paste the summary table into the reply — it is also written to
   `../.<repo>-verify-delta/summary.md`.

7. **Grep for closed issue numbers.** For every issue the PR closes, `grep -rn "#<n>\b" kornia/`
   in the touched modules: a surviving reference in a docstring or warning list means the change
   is incomplete (kornia#3999 shipped three docstrings promising behaviour it removed).

8. **Push once, then check it is actually being tested.** Push commits only — never
   `git push --tags` or `--follow-tags`, which would leak the round's `prefix-<round>` tag to the
   remote. After the push: `gh pr view <n> --json mergeStateStatus,statusCheckRollup`. `DIRTY`
   means no test workflow will run at all — GitHub runs nothing on an unmergeable PR, and the last
   green is stale. Test check-runs must *exist* for the new head, not merely be green.

9. **After two incremental rounds, ask for one full fresh review** of the whole branch with no
   prior context. Incremental reviews anchor on the last delta and miss interaction effects.

## Red flags

| Thought | Reality |
|---|---|
| "The reviewer asked for it, so it is in scope" | Findings expose defects; they do not authorise additions. Follow-up issue. |
| "The fix is a one-liner, no need to re-run the bar" | The one-liner in #4006 wave 9 keyed promotion off the wrong dtype. Run it. |
| "I'll add the rule to AGENTS.md while I'm here" | That bullet cost #4028 two rounds. Separate PR. |
| "257 is the boundary case" | Sweep `unrepresentable_sizes`; which operand rounds is unknown. |
| "The finding came with numbers, it must be right" | Re-measure on main. Refute in writing if it reproduces there. |
| "CI is green" | Green from which head? Check the run exists for the current SHA and the PR is not DIRTY. |
| "MPS/half is not in CI so it is not my problem" | It is documented as supported. `verify-delta` runs it. |

## Related

`kornia-precision-testing` for the test rules and helpers; `kornia-developer` for compile-first
changes; `TESTING.md` for the helper reference.
