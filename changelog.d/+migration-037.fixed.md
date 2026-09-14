`CameraModelBase.__init__` now validates `params` against the shape it documents, instead of storing
whatever it is given. The typed constructors (`PinholeModel`, `BrownConradyModel`, `KannalaBrandtK3`,
`Orthographic`) each apply the same two comparisons, so only the direct-construction path -- which is
public, and is what the class docstring's own example uses -- was unguarded. A short vector used to
construct and then fail with an `IndexError` from inside `AffineTransform.distort`, naming neither
`params` nor the camera; a rank-3 `(B, 1, N)` tensor used to construct, project, and silently return a
`(1, 1, 2)` result. Both now raise `ValueError` from the constructor. (#4316, #4369)
