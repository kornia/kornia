`spatial_gradient3d(..., order=2)` and `SpatialGradient3d(order=2)` now scale the mixed `dxy`, `dyz` and `dxz`
channels by 1/4. Previously, these channels were four times too large relative to `dxx`, `dyy` and `dzz`, so mixed
derivative values now change.
