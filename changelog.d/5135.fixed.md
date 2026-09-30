`line_segment_transfer_error_one_way` now returns the mean perpendicular distance of the mapped
image-1 endpoints from the image-2 line, in pixels. Before, the image-2 line was not normalised,
so the value was that distance multiplied by the image-2 segment length: the same 3-px offset
scored 197 on a 66-px segment and 994 on a 331-px one, and no single `inl_th` treated short and
long segments alike. Fixes #4867.
