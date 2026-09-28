`reproject_disparity_to_3D` (the `StereoCamera` method and the module-level function, which share
one body) no longer transposes the two pixel indices. The pixel meshgrid was unbound as
`v, u = torch.unbind(uv, dim=-1)`, but `create_meshgrid(normalized_coordinates=False)` returns
`(x, y)`, so the column fed `v` and the row fed `u`: `X` was computed from the row and `Y` from
the column. Every returned point was the value belonging to its transposed pixel -- kornia's
answer at `(row, col)` was what `cv2.reprojectImageTo3D` puts at `(col, row)`, the OpenCV
semantics this function was added to provide (#2042). A square rig with `fx == fy` and
`cx == cy` agrees only on the diagonal, so the error was silent rather than absent. Output
changes at every pixel with `row != col`, square inputs included. The real-data regression
fixture stored ten points as ten rows of one column while its own comment and its ground truth
describe one row of ten columns,
which is why it passed against the swapped code; it is now laid out as the comment says, and
the fixed code reproduces the unchanged ground-truth values. #4317's two wart pins for this defect
are retired and its strict-xfail convention test is now an ordinary passing regression. (#4269, #4366)
