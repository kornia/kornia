Validate the chroma plane height in `yuv422_to_rgb`, so a chroma plane whose height does not match the luma now
raises `ShapeError` instead of reaching `torch.cat` and dying there with a bare `RuntimeError` (#4050). 4:2:2
subsamples width only, so chroma keeps the full luma height; the guard checked only the width ratio, where the
4:2:0 twin `yuv420_to_rgb` checks both axes. This is observable to callers on the error path: `ShapeError` derives
from `BaseError`, not `RuntimeError`, so code catching `RuntimeError` around `yuv422_to_rgb` to handle a malformed
chroma plane will stop catching it. No previously working input changes behaviour — `torch.cat` already rejected
every mismatched height. A zero-width chroma plane whose height *also* differs now reaches the guard, since the new
clause short-circuits the width division; the matching-height case still raises
`ZeroDivisionError` and remains open as #4056.
