`PinholeCamera.from_parameters` now fills `height` and `width` for the whole batch. It built them with
`height_tmp[..., 0] += height` on a `(B,)` zero tensor, so every batch element after the first kept `0`
while `fx`, `fy`, `cx`, `cy`, `tx`, `ty` and `tz` were broadcast correctly, and the camera looked healthy
until something read its image size. `batch_size=1`, the only case that worked, is unchanged. (#4279)
