`pixel2cam` now rejects depth tensors outside the documented `Bx1xHxW` shape. The guard's predicate
bound as `(ndim != 4) and (shape[1] == 1)`, so four-dimensional multi-channel depth bypassed the
channel check — a three-channel depth silently broadcast into separate camera coordinates — while
scalar and one-dimensional inputs raised `IndexError` from the guard itself. Code that passed those
shapes and relied on them being accepted now raises `ValueError`. Valid `Bx1xHxW` depth is unaffected
and its output is unchanged. (#4314)
