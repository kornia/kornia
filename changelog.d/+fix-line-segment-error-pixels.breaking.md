The values `line_segment_transfer_error_one_way` returns change by a factor of 1 / (the image-2
segment length): the same geometric configuration now scores smaller on a longer segment. Any
threshold tuned to the old length-scaled values — a direct comparison with `inl_th**2`, or the
`soft_inl_th` of a hand-rolled iterated reweighting — must be re-tuned in pixels. The RANSAC
`homography_from_linesegments` score and polisher already used the pixel distance, so their
results are unchanged. Refs #4867.
