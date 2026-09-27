`fit_line` with `weights` now centres `D >= 3` points on the weighted centroid, as the `D = 2` branch does. A point
with weight 0 no longer moves the origin or tilts the direction. (#5014)
