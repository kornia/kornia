`solve_cubic` and `solve_quartic` return finite gradients for a row whose neighbours in the
same batch take a different branch. The `D > 0` branch selected its rows with `abs(R) > 1e-16`
alone, so it evaluated `sqrt(D)` on `D < 0` rows too. Those rows are never read back, but the
resulting `-Q / nan` stays in the graph and its backward returns `nan`, which reaches every
coefficient: a three-real-root cubic differentiated correctly on its own and gave `nan` as soon
as a one-real-root cubic shared the batch, and `solve_quartic` inherited it through its
resolvent cubic. The forward values are unchanged. (#4334, #4338)
