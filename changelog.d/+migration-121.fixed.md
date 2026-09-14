Fix `find_essential` and `run_5point` raising `TypeError: all() received an invalid combination of arguments` on
PyTorch before 2.2 (#4043). `run_5point`, `_solve_2x2_tikhonov_safe` and the scripted Nister helper each reduced
over two axes at once with `Tensor.any(dim=(-2, -1))` / `Tensor.all(dim=(-1, -2))`, and multi-dimension `any`/`all`
only landed in PyTorch 2.2, so the whole five-point solver was unusable on the declared `torch>=2.0` floor. The
reductions now flatten the two trailing axes first, which is equivalent on every version.
