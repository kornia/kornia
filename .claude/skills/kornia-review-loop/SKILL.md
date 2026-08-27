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

1. **Record the pre-fix SHA under a name only this round can own.**

   ```bash
   PREFIX_TAG="prefix-r<round>-$(git rev-parse --short HEAD)"
   git tag "$PREFIX_TAG"        # no -f: a name that already exists already points where you want
   ```

   A local tag, not a `/tmp` file, so it survives a shell restart without leaving scratch state in
   the kornia tree. The short SHA is not decoration: **tags are shared by every worktree of a
   repo**, so a bare `prefix-1` is a global name, and a second review running `git tag -f prefix-1`
   in another worktree silently repoints your baseline — after which step 4 measures "fails at the
   pre-fix SHA" against somebody else's branch. Never `-f` a name you did not create in this round.
   Never push it — `git push --tags` would leak it (`--follow-tags` would not: `git tag` makes a
   lightweight tag and `--follow-tags` pushes only annotated ones) — and `git tag -d "$PREFIX_TAG"`
   once the round is answered.

2. **Triage every finding before writing code.** For each one, reproduce it on the branch AND at
   the merge-base with **the ref this PR actually targets** — which is not always `main`:

   ```bash
   BASE_REF=$(gh pr view --json baseRefName -q .baseRefName)   # e.g. main, or a parent PR's branch
   git fetch origin "$BASE_REF"
   BASE=$(git merge-base "origin/$BASE_REF" HEAD)
   ```

   `origin/main` answers "is this broken today?"; the merge-base answers "did *this branch* break
   it?", and only the second is a triage verdict. Once `main` moves past the branch point the two
   diverge in both directions: a regression the branch introduced can reproduce on `main` as well
   (someone else's change), and step 2 files it "pre-existing — do NOT fix", the exact false triage
   this step exists to prevent; and if the PR's own work has already landed on `main` by another
   route, the probe reads the fix back as the baseline (this happened to two runs in the #4035
   evaluation and both had to be rerun).

   Hard-coding `main` is a second, separate error on a **stacked or release-target PR**. Measured
   on #4035, which targets `feat/testing-precision-helpers`: `merge-base origin/main HEAD` sits 23
   commits back and drags 9 files belonging to the parent PR into the delta, so every finding
   against the parent's code triages as "this branch broke it" and the gate below measures a delta
   nobody asked for. `merge-base "origin/$BASE_REF" HEAD` is 10 commits back and covers exactly the
   files under review. Reserve `origin/$BASE_REF` for step 6's tip-vs-tip gate; every step-2
   finding probe uses the merge-base `$BASE` above, including a non-pytest reproduction.

   The cheapest correct probe reuses the tool, which already creates the worktree, guards the
   import, and diffs failing-test sets — which is the triage question:

   ```bash
   pixi run verify-delta -- --base "$BASE" --tests tests/<module> --only "cpu float32"
   ```

   A `NEW [<surface>] <id>` line is a *regression*. A finding's test inside `unchanged` is
   **not yet** a pre-existing verdict: `verify-delta` diffs failing-test *ids*, not failure
   *causes*, so a test that failed at the base for reason A and fails on the branch for a new
   reason B lands in `unchanged` under one id and the row reads `0 new`. Before filing anything
   "pre-existing", read the two failures and confirm they are the same failure — same exception
   type, same assertion, same first traceback frame in the library. The set diff is the coarse
   gate; the verdict needs the observed behaviour. One worktree serves the whole round
   (`../.<repo>-verify-main`, re-pointed with `--main-worktree`), so do not create a second
   `../main-probe` beside it. `verify-delta` refuses to run on a dirty checkout, so commit or stash
   the round's work in progress before probing (or pass `--allow-dirty` and say so in the reply —
   its verdict then describes your working tree, not HEAD).

   When the repro is not a pytest run — a REPL snippet or a script — run the **exact repro** on
   the branch and the `$BASE` worktree, rather than substituting a nearby pytest invocation. Check
   what each tree imported before believing either answer:

   ```bash
   BRANCH_WT=$(git rev-parse --show-toplevel) || exit $?
   PIXI_ENV=${PIXI_ENVIRONMENT_NAME:-default}  # or set this to the environment used by the failing test
   PY=$(cd "$BRANCH_WT" && pixi run -e "$PIXI_ENV" uv run python -c 'import sys; print(sys.executable)') || exit $?
   PROBE_WT="$(dirname "$BRANCH_WT")/.$(basename "$BRANCH_WT")-verify-main"  # made at $BASE
   run_repro() {
       local tree=$1
       (cd "$tree" \
        && PYTHONPATH=$PWD "$PY" -c 'import kornia, pathlib, sys; p = pathlib.Path(kornia.__file__).resolve(); sys.exit(0 if p.is_relative_to(pathlib.Path.cwd()) else f"kornia imported from {p}")' \
        && PYTHONPATH=$PWD "$PY" -c '<paste the exact REPL snippet here>')
       # For a script, replace only the final command with: PYTHONPATH=$PWD "$PY" path/to/repro.py <args>
   }
   run_repro "$BRANCH_WT"
   run_repro "$PROBE_WT"
   ```

   `sys.executable` from the project task's `uv run` environment, not bare
   `python` or a hard-coded `.venv/bin/python`: `pixi.toml` points the `py312` and `py313`
   features at `.venv-py312` and `.venv-py313` (`UV_PROJECT_ENVIRONMENT`), so under
   `pixi run -e py312` that literal path is either absent or a *different* Python and torch build —
   and the rounding question you are probing is answered by the build, not by the source. It is also
   what makes the line work unchanged on Windows, where the interpreter is `Scripts\python.exe`, and
   it is what `verify_delta._assert_imports_from` uses for the same reason.

   Keep the guard on one line — `python -c` rejects an indented continuation with
   `IndentationError`, which exits non-zero and reads as "the guard fired". Do not set
   `PYTHONPATH` and stop there: the `.venv` is shared across these worktrees and its editable
   install points at whichever tree installed last, so the same command *without* `PYTHONPATH`
   answers from that tree instead — measured here, `cd $PROBE_WT && "$PY" -c "import kornia"` printed
   the `kornia/__init__.py` of a *different* checkout than the one it was standing in. Whether
   `PYTHONPATH` wins depends on how the editable install was built (a `.pth` path entry loses to
   it; an `__editable___*` meta-path finder does not, being consulted before `sys.path` at all), so
   assert the answer rather than reasoning about the ordering. Get it wrong
   and the probe reports the *branch* while you record a merge-base measurement, and the verdict
   flips from regression to pre-existing with nothing visible to show for it — which is why
   `verify_delta._assert_imports_from` exists.

   Four outcomes:
   - *regression* (fails on branch, passes at the merge-base) → fix it;
   - *pre-existing* (fails on both) → reply with both measurements, file or link an issue, do
     NOT fix it in this PR;
   - *wrong* → refute in writing with the measurement;
   - *reproduces, diagnosis wrong* → the symptom is real and the finding's stated cause, condition
     or "Suggested fix" is not. Fix the symptom from your own measurement, and say which part of
     the finding you are not adopting.

   The fourth verdict exists because the first three only ask whether the *symptom* reproduces.
   A finding's diagnosis and its suggested fix are claims to verify, not instructions to execute.
   Whenever a finding states a **condition** ("requires X", "only when X"), a **cause** ("because
   X"), a **threshold** ("from N on") or a **property** its suggested change is said to keep
   ("still catches every partial fix", "loses nothing"), run the two falsifying probes before
   writing any of it into the tree: one instance that satisfies X and still fails, and one that
   violates X and still succeeds. For a property, search A is an instance the finding says the
   change still handles and the change does not. Paste both outputs under the verdict line. The
   pattern this guards against, measured on kornia#3984 and again in the skill evaluation: a
   finding's *figures* are re-measured and match, which feels like verification, and its
   *diagnosis* is then adopted untested; the historical round and both evaluation arms shipped a
   reviewer-supplied condition that four probe lines would have refuted, and a reviewer-supplied
   cause reached a public docstring the same way.

   Write every verdict into the reply as its own fenced block, so a finding you never answered is
   visibly absent instead of quietly missing:

   ```text
   ### Triage: <finding, abridged>
   - symptom on branch / at merge-base: <command → output> / <command → output>
   - verdict: regression | pre-existing | wrong | reproduces, diagnosis wrong | reproduces, diagnosis confirmed
   - stated condition/cause/threshold/property: <"X" quoted from the finding | none stated>
   - counterexample search A — an input that SATISFIES X and still FAILS: <inputs tried (≥2, none from the finding) → outputs> → found | not found
   - counterexample search B — an input that VIOLATES X and still SUCCEEDS: <inputs tried (≥2, none from the finding) → outputs> → found | not found
   - stated set (consumers, sites, cells, files): <list quoted from the finding | none stated>
   - tree-wide search for that set: <`git grep`/`--collect-only` command → output> → finding's list complete | incomplete: <what it misses>
   ```

   The two searches are attempts to *break* the diagnosis, not to illustrate it. The finding's
   own examples are consistent with its diagnosis by construction, so they do not count; construct
   new inputs that isolate X from everything else the finding's examples vary (the same matrix
   with one row changed, exact integers at a large power of two, a shape the finding never tried).
   If either search finds an instance, the verdict is "diagnosis wrong" and X does not enter the
   tree. A *found* instance ends the change; it is not a caveat inside it. The reply declines, or
   narrows the change to what the instance leaves standing, and carries the instance. Shipping the
   suggested change with a comment that discloses the gap is not one of the four outcomes: it
   records that the probe ran and adopts the refuted premise anyway. In the evaluation one run
   monkeypatched the partial fix the finding said could not escape its reduced matrix, watched it
   escape, wrote "an inherent tradeoff of the reduction", and shipped the reduction.

   No class of finding waives the block. A finding that reads as organisational — consolidate,
   deduplicate, move, reorder — still states facts to search: the **set** it names (which modules
   consume the constant, how many skip sites, how many cells) and the **property** the
   consolidation keeps. For a set the search is one tree-wide `git grep` or `--collect-only`, and
   its output is the list that ships; the finding's list never does, however complete it looks.
   On kornia#3942 a reviewer-supplied consumer list went into the canonical docstring three files
   short of the grep, and the next round found the gap. "Diagnosis confirmed" is only available
   with both searches run on new inputs and both `not found`; matching the finding's figures is
   not a probe of its diagnosis. In the evaluation,
   one run skipped both searches and wrote "confirmed — fix applied as diagnosed", and the next
   filled them with the finding's own two examples and did the same, for a diagnosis that one
   modified matrix refutes. A finding is never dropped silently; the next
   reviewer re-raises anything unanswered. Findings that arrive with confident numbers are still re-measured — one on #4006
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
   already-fixed tree, passes, and you wrongly conclude it is vacuous. Use the step-1 tag and the
   self-contained probe below. It derives its path from the repository root, so it is safe even
   when invoked from a subdirectory, and gives each tag a distinct worktree:

   ```bash
   REPO_WT=$(git rev-parse --show-toplevel) || exit $?
   PIXI_ENV=${PIXI_ENVIRONMENT_NAME:-default}  # or set this to the environment used by the failing test
   PY=$(cd "$REPO_WT" && pixi run -e "$PIXI_ENV" uv run python -c 'import sys; print(sys.executable)') || exit $?
   PROBE_WT=$(mktemp -d "$(dirname "$REPO_WT")/.$(basename "$REPO_WT")-prefix-probe-XXXXXX") || exit $?
   rmdir "$PROBE_WT" || exit $?  # reserve a collision-free name, then let git create the worktree
   PROBE_CREATED=0
   cleanup_prefix_probe() { [ "$PROBE_CREATED" -eq 1 ] && git -C "$REPO_WT" worktree remove --force "$PROBE_WT" 2>/dev/null || true; }
   trap cleanup_prefix_probe EXIT HUP INT TERM
   git -C "$REPO_WT" worktree add --detach "$PROBE_WT" "$PREFIX_TAG" || exit $?
   PROBE_CREATED=1
   # Copy <test> and every new module it imports into the same relative paths in "$PROBE_WT".
   (cd "$PROBE_WT" \
    && PYTHONPATH=$PWD "$PY" -c 'import kornia, pathlib, sys; p = pathlib.Path(kornia.__file__).resolve(); sys.exit(0 if p.is_relative_to(pathlib.Path.cwd()) else f"kornia imported from {p}")' \
    && PYTHONPATH=$PWD "$PY" -m pytest <test>)
   # Inspect the failure: it must be the target assertion, not collection/import failure.
   ```

   The active `$PY` and `$PROBE_WT` are assigned in the same shell that creates and probes the
   worktree; the guarded trap removes it only after this shell created it, when the expected
   assertion failure ends the command. Copy the test file over (new or modified — it overwrites
   the tag's version)
   **together with every module it now imports that the tag does not have** (a new `testing/`
   helper, a new fixture, a new `_historical.py`), then run the shown command with its import guard
   in front — the shared `.venv` misdirects this probe the same way, and the
   probe worktree has no `.venv` of its own, so a bare `python` or `uv run` there would answer the
   rounding question under a different interpreter and torch build. Confirm it fails **with the
   assertion the fix is about**. A collection error or `ImportError` is *not* that
   failure: it means the test never ran at the tag, and counting it as "it fails on the pre-fix
   SHA" green-lights a test that has never been executed against the bug. Read the failure line
   (`AssertionError`, `Failed: DID NOT RAISE`), never a bare non-zero exit code.

5. **Self-review the delta, and ledger every claim it makes.** Two fenced sections go into the
   reply, filled in. Neither is optional; in 10 of 12 evaluation runs the review was skipped when
   it was only described, and performed in 5 of 5 once the section had to be pasted.

   **(a) Delta self-review.** When the delta touches code under `kornia/` or test *logic* (not only
   comments and docstrings), dispatch a fresh subagent on `git diff "$PREFIX_TAG"..HEAD` **together
   with the finding's text**, with this instruction: "The reviewer's diagnosis is a hypothesis, not
   a fact. Attack only the new code. For every changed line check: is a size cast into a half dtype
   before a division; does a degenerate/empty path validate less or more than the full path; is a
   `dtype=None` resolved before promotion is decided; does a new compile test carry
   `dynamo`/`compile` in its name; do MPS/half skips match the nearest existing test; did a
   docstring, comment, or changelog sentence next to the change stop being true; does the fix in
   one function need mirroring in its 3d/other sibling; and does any sentence the diff adds assert
   a mechanism, cause, condition or threshold that no command in this round executed — including
   one the reviewer supplied; and what in the diff can change elsewhere in the tree without any
   test in the diff failing? For the mechanism item, run the counterexample; do not read for it." Fix
   what it finds. Repeat once if it found anything. For a comments-and-docstrings-only delta,
   answer the same questions yourself — the subagent is the expensive part and the pasted section
   is the part that works.

   ```text
   ### Delta self-review ($PREFIX_TAG..HEAD)
   - size cast into a half dtype before a division: <none | file:line + outcome>
   - degenerate/empty path validates less or more than the full path: <…>
   - dtype=None resolved before promotion is decided: <…>
   - new compile test carries dynamo/compile in its name: <…>
   - MPS/half skips match the nearest existing test: <…>
   - docstring/comment/changelog next to the change still true: <…>
   - fix mirrored in the 3d/other sibling, and the sibling's accepted calls still pass: <…>
   - a mechanism/cause/condition/threshold asserted that this round did not execute: <none | which, and the probe that settled it>
   - goes stale without failing any test in this diff (version string, default, count, list of sites): <none | which → pinned by <test/probe> | unpinned, said so in the reply>
   ```

   The first seven lines look inward at the code and catch incompleteness (in the evaluation they
   turned one arm's first draft, which matched the control arm's answer, into the complete one).
   They do not catch a premise inherited from the reviewer — the same section confirmed a wrong
   condition in the evaluation — which is what the eighth line and the ledger below are for.

   The ninth line looks at the *result* rather than the diagnosis: after the diff is written, name
   one thing in it that can go stale on its own because nothing in the diff fails when it changes
   — a torch version in a skip reason, a default read from another module, a count, a list of
   sites. Pin it (a runtime probe in place of the version string, an assertion on the default, a
   grep-derived list) or write in the reply that it is unpinned and where it lives. Every step-2
   search is on the diagnosis; without this line nothing asks what the fix itself now depends on.
   Both evaluation arms consolidated three MPS skips into one reason string and kept a torch
   version number in it that gates nothing — the sentence the next round raised.

   **(b) Claims ledger.** One row per sentence the delta *adds or moves* — docstring, comment,
   changelog, test comment, reply — that asserts a **mechanism** (why), a **cause** (because), a
   **condition** (only when / requires / whenever), a **threshold** (from N on / at k) or a
   **count/set** ("one site", "N cells", "every consumer: A, B, C"). A sentence you move is a
   sentence you write: this round makes it the canonical copy, so every figure in it is
   re-derived here, not inherited. A consolidation that promotes an unexecuted figure into the one
   place everyone reads is how a wrong number outlives the round that could have caught it
   (kornia#3942 carried "loses about seven digits" into the canonical copy; it was off by more
   than a digit). A refactor has rows too: the sentence saying what it consolidated is a count.

   ```text
   ### Claims ledger ($PREFIX_TAG..HEAD)
   | claim (file:line) | kind | executed by | scope (dtype · device · sizes · torch build) | source |
   | <sentence, abridged> | figure | mechanism | cause | condition | threshold | <see below> | <…> | measured | read <file:line> |
   ```

   One sentence can be several rows: "returns `0.0` *because* the value is below the smallest
   subnormal" is a `figure` row and a `cause` row, and the cause needs its own execution. What
   `executed by` must contain depends on `kind`:

   - `figure` — the command whose output the sentence states;
   - `mechanism` / `cause` — a probe that distinguishes it from the alternative, not the result it
     explains: a bitwise-equal result does not execute "no pivot swap occurs", and a `0.0` does
     not execute "because it is below the subnormal";
   - `condition` — counterexample searches A and B from the triage block, both `not found` on
     inputs the finding did not supply;
   - `threshold` — the bisect: the smallest `k` at which the behaviour flips, with the value on
     each side. A pin that backs a threshold asserts at the boundary, not at the example;
   - `count`/`set` — the tree-wide `git grep` or `--collect-only` whose output *is* the list or the
     number. A list taken from the finding has source "the reviewer said so" and does not ship.

   `source` is `measured` (this round ran it at that exact literal, size, dtype and device) or
   `read` (a claim about code, with the file:line you opened). A row whose only source would be
   *the reviewer said so* or *it is plausible* does not ship: run it, or delete the sentence.
   Checking other dtypes at the example size is not a bisect (kornia#3984 promoted a reviewer's example size to a threshold
   that was several powers of two off; the evaluation's with-arm did the same with
   `kornia-precision-testing` loaded — the rule is stated here because it did not transfer from
   the test register on its own).

   A reply without both sections is an unanswered round.

6. **Gate: `pixi run verify-delta`.** Zero `new` failures on every available surface — the four
   `--only` names are `cpu float32`, `cpu float16,bfloat16,float64`, `mps float32`, and
   `inductor cpu float32` (runs `-k "dynamo or compile"`). Half precision and MPS have no CI job —
   this is their only signal. With no flags it derives test dirs from the diff, and a change under
   `testing/`, `conftest.py`, `pyproject.toml`, or `pixi.toml` widens to the whole suite (~20 min
   per surface per tree); scope it while iterating with
   `pixi run verify-delta -- --tests tests/geometry --only "cpu float32"`. On a PR that does not
   target `main`, add `--base "origin/$BASE_REF"` from step 2 — the gate asks "does the tip break
   the branch it merges into", so it wants that branch's *tip*, where step 2's triage wanted the
   *merge-base* with it. The tool refuses a dirty checkout by default: its automatic scope is
   `base...HEAD`, so an uncommitted fix would make a broken, pushable HEAD read green. If you
   intentionally gate work in progress, pair `--allow-dirty` with explicit `--tests`; automatic
   selection cannot discover working-tree-only paths.

   Exit 0 = no new failures, 1 = new failures (the `NEW [<surface>] <id>` lines name them), 2 = a
   selected, available surface was never measured — a failure, not a pass. **Exit 0 is necessary
   and not sufficient: read the table.** Every row you selected and this machine can run must show
   numbers. The three non-numeric cells are distinct and only one of them is acceptable in a green
   gate:

   - `not selected` — excluded by `--only`. Fine, but if that leaves half precision or MPS unrun,
     say so in the reply; they have no CI job, so nobody else will run them.
   - `unavailable` — the machine has no such backend (MPS off a Mac). Name it in the reply.
   - `unverified` — selected, available, and it did not produce a measurement. Resolve and rerun
     it; do not push on it. This is the exit-2 case.

   A diff that maps to no test target at all (a docs-only PR) exits 0 and prints "nothing to
   verify" — that is not a green gate either. The hand gate for such a delta is bounded, not the
   four-surface sweep: `pixi run doctest` on the touched module, plus the companion pins of every
   sentence you changed, run at exactly the dtypes and devices those sentences name (a sentence
   about float16 is gated on float16; one that names no dtype is gated on cpu float32). State in
   the reply which of these you ran and their result. A `N*` row means the base revision ran
   fewer of the test paths than the branch did (a new test package it does not have at all, or
   only some of them), so failures under a path it never ran count as new unconditionally. Paste the summary table into the reply — it is also written to
   `../.<repo>-verify-delta/summary.md`.

7. **Grep for closed issue numbers.** For every issue the PR closes, run the `AGENTS.md` rule —
   "before merging a change that closes one of those issues, run
   `grep -rnE "#NNNN|issues/NNNN" kornia/ tests/`" — not a bare `#<n>` grep of `kornia/`: the
   `#`/`issues/` anchors keep the pattern off float literals, and the `tests/` half finds the pins.
   A surviving hit in `kornia/` means the change is incomplete (kornia#3999 shipped three docstrings
   promising behaviour it removed); a hit in `tests/` is a pin to re-check in the same change, not
   by itself a defect. `AGENTS.md` carries the full list of wordings the number hides behind.

8. **Push once, then check it is actually being tested.** Push commits only — never
   `git push --tags`, which would leak the round's `$PREFIX_TAG` to the remote.
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
| "257 is the boundary case" / "it underflows from the reviewer's example size on" | An example is not a boundary until bisected. In a test, sweep `unrepresentable_sizes`; in prose, find the smallest `k` where it flips (step 5b). |
| "The finding came with numbers, it must be right" | Re-measure at the merge-base with the PR's own base ref. Refute in writing if it reproduces there. |
| "The reviewer's diagnosis explains the numbers, and the suggested fix follows from it" | The numbers were measured; the diagnosis was not. Run the two falsifying probes (step 2) before the condition or cause enters the tree. |
| "The base branch is `main`" | `gh pr view --json baseRefName`. A stacked PR measured against `main` gates its parent's diff too. |
| "The test already fails at the base, so it's pre-existing" | `verify-delta` diffs failure *ids*. Same id, new cause, still reads `unchanged`. Compare the two failures. |
| "verify-delta exited 0" | Read the table. An `unverified` row is a hole; `not selected` and `unavailable` rows need naming in the reply. |
| "CI is green" | Green from which head? Check the run exists for the current SHA and the PR is not DIRTY. |
| "MPS/half is not in CI so it is not my problem" | It is documented as supported. `verify-delta` runs it. |
| "The sibling has the same typo, I'll fix both" | Probe the sibling's accepted calls first; the 3d guard's `or` was load-bearing for unbatched input. |
| "A docs-organization finding — nothing to falsify" | It names a set. One tree-wide grep; its output replaces the finding's list (step 2). |
| "The probe found a gap, but that is inherent to the suggested change; I'll disclose it in a comment" | The finding claimed there was no gap. Found = diagnosis wrong = decline or narrow. A disclosure line is adoption. |
| "Pure refactor, nothing to ledger" | "One site", "N cells", "every consumer" are counts. Run the command that counts them and ledger it (step 5b). |
| "That figure was already in the tree; I only moved it" | Your sentence is now the canonical copy. Re-derive every figure in it (step 5b). |
| "Every test passes, the refactor is done" | Name what in the diff goes stale without failing a test — a version string, a default, a count — and pin it or say it is unpinned (step 5a, last line). |

## Related

`kornia-precision-testing` for the test rules and helpers; `kornia-developer` for compile-first
changes; `TESTING.md` for the helper reference.
