`solve_cubic`'s `D <= 0` (three-real-roots) branch no longer returns `nan` gradients for a
repeated or near-repeated real root (#4290, #4299). It differentiates `acos(R / sqrt(-Q3))`, whose own
derivative `-1/sqrt(1-x^2)` is unbounded at `x = +-1`; the branch condition guarantees the ratio
lies in `[-1, 1]`, but a repeated or near-repeated root pushes it to exactly that boundary,
where the value is correct but the derivative diverges. A double-root quartic reaches this
through `solve_quartic`'s resolvent cubic routinely, not as an edge case: measured on 20,000
random real-rooted quartics, 10.7% hit a non-finite gradient or a disconnected graph overall,
35.3% for exact double-root pairs specifically. Same failure shape as the `acos`/`asin`
boundary in `quaternion_exp_to_log`/`euler_from_quaternion` (#4007, fixed in #4228) — this is a
different call site, not covered by that fix, using the same guard: differentiate a substituted
safe argument, take the value from a detached copy of the real (possibly boundary) argument.
Forward values are unchanged; re-running the same 20,000-trial search with the fix applied
drops the failure count to 1,269 — a real, separate residual (a disconnected-graph case with
two simultaneous double-root pairs) remains and is tracked in #4290, not fixed here.
