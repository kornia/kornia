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
   `git push --tags` would leak it to the remote (`--follow-tags` would not: `git tag -f` makes a
   lightweight tag and `--follow-tags` pushes only annotated ones) — and delete it
   (`git tag -d prefix-<round>`) once the round is answered.

2. **Triage every finding before writing code.** For each one, reproduce it on the branch AND at
   the branch's **merge-base** — not at `origin/main`:

   ```bash
   BASE=$(git merge-base origin/main HEAD)
   ```

   `origin/main` answers "is this broken today?"; the merge-base answers "did *this branch* break
   it?", and only the second is a triage verdict. Once `main` moves past the branch point the two
   diverge in both directions: a regression the branch introduced can reproduce on `main` as well
   (someone else's change), and step 2 files it "pre-existing — do NOT fix", the exact false triage
   this step exists to prevent; and if the PR's own work has already landed on `main` by another
   route, the probe reads the fix back as the baseline (this happened to two runs in the #4035
   evaluation and both had to be rerun). `pixi run verify-delta`'s `--base origin/main` default is
   deliberately different — a regression *gate* asks "does the tip break the merge target", which
   is the tip-vs-main question. The two steps differ on purpose.

   The cheapest correct probe reuses the tool, which already creates the worktree, guards the
   import, and diffs failing-test sets — which is the triage question:

   ```bash
   pixi run verify-delta -- --base "$BASE" --tests tests/<module> --only "cpu float32"
   ```

   A `NEW [<surface>] <id>` line is a *regression*; a finding's test inside `unchanged` is
   *pre-existing*. One worktree serves the whole round (`../.<repo>-verify-main`, re-pointed with
   `--main-worktree`), so do not create a second `../main-probe` beside it.

   When the repro is not a pytest run — a REPL snippet, a script — run it in that same worktree,
   and check what it imported before believing the answer:

   ```bash
   PROBE=../.kornia-verify-main   # ../.<checkout-dir-name>-verify-main, the worktree just made at $BASE
   (cd "$PROBE" \
    && PYTHONPATH=$PWD "$OLDPWD/.venv/bin/python" -c \
       'import kornia, pathlib, sys; p = pathlib.Path(kornia.__file__).resolve(); sys.exit(0 if p.is_relative_to(pathlib.Path.cwd()) else f"kornia imported from {p}")' \
    && PYTHONPATH=$PWD "$OLDPWD/.venv/bin/python" -m pytest <test> -q)
   ```

   Keep the guard on one line — `python -c` rejects an indented continuation with
   `IndentationError`, which exits non-zero and reads as "the guard fired". Do not set
   `PYTHONPATH` and stop there: the `.venv` is shared across these worktrees and its editable
   install points at whichever tree installed last, so the same command *without* `PYTHONPATH`
   answers from that tree instead — measured here, `cd $PROBE && python -c "import kornia"` printed
   the `kornia/__init__.py` of a *different* checkout than the one it was standing in. Whether
   `PYTHONPATH` wins depends on how the editable install was built (a `.pth` path entry loses to
   it; an `__editable___*` meta-path finder does not, being consulted before `sys.path` at all), so
   assert the answer rather than reasoning about the ordering. Get it wrong
   and the probe reports the *branch* while you record a merge-base measurement, and the verdict
   flips from regression to pre-existing with nothing visible to show for it — which is why
   `verify_delta._assert_imports_from` exists. (Windows: `.venv/Scripts/python.exe`.)

   Three outcomes:
   - *regression* (fails on branch, passes at the merge-base) → fix it;
   - *pre-existing* (fails on both) → reply with both measurements, file or link an issue, do
     NOT fix it in this PR;
   - *wrong* → refute in writing with the measurement.

   Write every verdict into the reply as its own line — `<finding> — regression | pre-existing |
   wrong — <the two measurements>` — so a finding you never answered is visibly absent instead of
   quietly missing. A finding is never dropped silently; the next reviewer re-raises anything
   unanswered. Findings that arrive with confident numbers are still re-measured — one on #4006
   attributed to the branch a divergence that was identical at the merge-base.

3. **Fix defects only.** If a finding makes you want to *add* something — a helper, a rule in
   `AGENTS.md`, a deprecation, a new test file for an adjacent function — write it as a follow-up
   issue and link it in the reply. Mid-PR additions are reviewed as fixes and generate rounds. The
   PR only shrinks or corrects during review. A same-class defect in a sibling function is in scope
   only after step 2's merge-base probe has been run on that sibling too; a guard you tighten
   without probing the merge-base may be removing behaviour the merge target deliberately
   preserves. Probe what the sibling currently *accepts* as well as what it rejects — enumerate
   its working calls at the merge-base (unbatched matrix, empty `dsize`, singleton axis) and keep
   one test per call. A guard tightened
   from `or` to `and` in `warp_perspective3d` passed every rejection probe and still broke the
   unbatched `eye(4)` call the merge-base accepted (kornia#4006 eval, both model tiers).

4. **Write tests that cannot pass vacuously.** Use the `kornia-precision-testing` skill for any
   test touching dtype, device, capture or degenerate sizes. Before trusting a new regression test,
   check it FAILS on the pre-fix SHA — `git stash; pytest <test>; git stash pop` is a no-op once
   the fix is committed (which step 5 presumes): there is nothing to stash, the test runs on the
   already-fixed tree, passes, and you wrongly conclude it is vacuous. Use the step-1 tag instead:
   `git worktree add --detach ../prefix-probe prefix-<round>`, copy the test file over (new or
   modified — it overwrites the tag's version) **together with every module it now imports that the
   tag does not have** (a new `testing/` helper, a new fixture, a new `_historical.py`), then run
   `(cd ../prefix-probe && PYTHONPATH=$PWD "$OLDPWD/.venv/bin/python" -m pytest <test>)`, prefixed
   with step 2's import guard — the shared `.venv` misdirects this probe the same way, and the
   probe worktree has no `.venv` of its own, so a bare `python` or `uv run` there would answer the
   rounding question under a different interpreter and torch build. Confirm it fails **with the
   assertion the fix is about**. A collection error or `ImportError` is *not* that
   failure: it means the test never ran at the tag, and counting it as "it fails on the pre-fix
   SHA" green-lights a test that has never been executed against the bug. Read the failure line
   (`AssertionError`, `Failed: DID NOT RAISE`), never a bare non-zero exit code. Then
   `git worktree remove --force ../prefix-probe` — the copied-in test file is untracked (new) or
   modified (pre-existing) there, so a plain `git worktree remove` refuses ("contains modified or
   untracked files").

5. **Self-review the delta with a fresh agent.** Dispatch a subagent on
   `git diff prefix-<round>..HEAD` with this instruction: "Attack only the new code. For
   every changed line check: is a size cast into a half dtype before a division; does a
   degenerate/empty path validate less or more than the full path; is a `dtype=None` resolved
   before promotion is decided; does a new compile test carry `dynamo`/`compile` in its name; do
   MPS/half skips match the nearest existing test; did a docstring, comment, or changelog sentence
   next to the change stop being true; does the fix in one function need mirroring in its 3d/other
   sibling." Fix what it finds. Repeat once if it found anything. Whether or not you could dispatch
   a subagent, the reply to the reviewer MUST include this section, filled in, verbatim:

   ```text
   ### Delta self-review (prefix-<round>..HEAD)
   - size cast into a half dtype before a division: <none | file:line + outcome>
   - degenerate/empty path validates less or more than the full path: <…>
   - dtype=None resolved before promotion is decided: <…>
   - new compile test carries dynamo/compile in its name: <…>
   - MPS/half skips match the nearest existing test: <…>
   - docstring/comment/changelog next to the change still true: <…>
   - fix mirrored in the 3d/other sibling, and the sibling's accepted calls still pass: <…>
   ```

   A reply without this section is an unanswered round. In 10 of 12 evaluation runs the review was
   skipped when it was only described; it is performed when the section has to be filled in.

6. **Gate: `pixi run verify-delta`.** Zero `new` failures on every available surface — the four
   `--only` names are `cpu float32`, `cpu float16,bfloat16,float64`, `mps float32`, and
   `inductor cpu float32` (runs `-k "dynamo or compile"`). Half precision and MPS have no CI job —
   this is their only signal. With no flags it derives test dirs from the diff, and a change under
   `testing/`, `conftest.py`, `pyproject.toml`, or `pixi.toml` widens to the whole suite (~20 min
   per surface per tree); scope it while iterating with
   `pixi run verify-delta -- --tests tests/geometry --only "cpu float32"`. Exit 0 = no new
   failures, 1 = new failures (the `NEW [<surface>] <id>` lines name them), 2 = nothing was
   verified (no surface ran pytest on the branch side — a typo'd `--tests` path or an unavailable
   `--only` surface), which is a failure, not a pass. A diff that maps to no test directory at all
   (a docs-only PR) also exits 0 and prints "nothing to verify" — that is not a green gate; state
   in the reply which surfaces you instead ran by hand. A `N*` row means the base revision ran
   fewer of the test paths than the branch did (a new test package it does not have at all, or only
   some of them), so failures under a path it never ran count as new unconditionally; `skipped`
   means that surface was deselected, unavailable, or could not be measured — a base tree where
   pytest exited without finishing is `skipped`, never a clean row. Paste the summary table into
   the reply — it is also written to `../.<repo>-verify-delta/summary.md`.

7. **Grep for closed issue numbers.** For every issue the PR closes, run the `AGENTS.md` rule —
   "before merging a change that closes one of those issues, run
   `grep -rnE "#NNNN|issues/NNNN" kornia/ tests/`" — not a bare `#<n>` grep of `kornia/`: the
   `#`/`issues/` anchors keep the pattern off float literals, and the `tests/` half finds the pins.
   A surviving hit in `kornia/` means the change is incomplete (kornia#3999 shipped three docstrings
   promising behaviour it removed); a hit in `tests/` is a pin to re-check in the same change, not
   by itself a defect. `AGENTS.md` carries the full list of wordings the number hides behind.

8. **Push once, then check it is actually being tested.** Push commits only — never
   `git push --tags`, which would leak the round's `prefix-<round>` tag to the remote.
   After the push: `gh pr view <n> --json mergeStateStatus,statusCheckRollup`. `DIRTY`
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
| "The finding came with numbers, it must be right" | Re-measure at the merge-base. Refute in writing if it reproduces there. |
| "CI is green" | Green from which head? Check the run exists for the current SHA and the PR is not DIRTY. |
| "MPS/half is not in CI so it is not my problem" | It is documented as supported. `verify-delta` runs it. |
| "The sibling has the same typo, I'll fix both" | Probe the sibling's accepted calls first; the 3d guard's `or` was load-bearing for unbatched input. |

## Related

`kornia-precision-testing` for the test rules and helpers; `kornia-developer` for compile-first
changes; `TESTING.md` for the helper reference.
