`PinholeCamera.project` now reports rank-1 inputs with the same explicit `ValueError` used by the point-conversion
helpers instead of an internal `IndexError`. Refs #4266. (#4450)
