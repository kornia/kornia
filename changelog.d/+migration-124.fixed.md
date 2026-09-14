Restore the YUV test coverage that #3539 deleted, and fix the `kornia.color` YUV docstring examples (#4045).
`rgb_to_yuv422` and `yuv422_to_rgb` both documented an example that called the 4:2:0 function instead of
themselves, so a reader copying either got the wrong conversion and the wrong chroma shapes; several other
examples stated a shape in a trailing comment that disagreed with what the function returns
(`RgbToYuv420` said `2x1x2x3` for a `(2, 2, 2, 3)` chroma plane). Every example in `kornia/color/yuv.py` now
asserts its output shape as a doctest instead of stating it in a comment. The 4:2:2 docstrings also claimed
the input only had to be divisible by 2 "vertical" while the guard rejects odd height *and* odd width, so a
reader padding one axis hit a `ShapeError`, and `yuv422_to_rgb` labelled its chroma argument "UV (luma)".
The four `ShapeError` messages those guards raise said "evenly disible by 2" and now say "divisible", as do
the two identical messages in `kornia/color/raw.py`. `yuv422_to_rgb`/`Yuv422ToRgb` also gained a
`.. warning::` naming a known defect: they validate only the chroma *width*, so a wrong chroma height
surfaces as a bare `RuntimeError` from `torch.cat` (#4050). The same entry gave every YUV-to-RGB form a
second warning, for the round trip that #4044 has since fixed; those blocks are gone, see the entry
above. On the test side, both gradcheck skips now cover the XLA/TPU fixture as well as MPS, since XLA
lowers a float64 request to float32, where gradcheck's default `eps=1e-6`
makes the numerical Jacobian invalid: the name-based marker in `conftest.py` for tests called `*gradcheck*`,
and the device guard in `BaseTester.gradcheck` for the six callers named something else. No runtime
behavior changed.
