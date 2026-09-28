`StereoCamera` now rejects projection matrices whose last two dimensions are not `(3, 4)` with a
`StereoException` naming the invalid camera. Previously, the shape guards never fired, so malformed
matrices could be accepted or fail later with an unrelated tensor error. (#4385)
