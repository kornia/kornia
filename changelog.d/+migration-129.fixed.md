Fix `laf_is_inside_image` treating the image extent as `(w, h)` rather than `(w - 1, h - 1)`, which made its
bounds asymmetric: the lower bound rejected anything left of `x = 0`, but the upper bound accepted `x = w`, a
full pixel past the last valid column `w - 1` (#4064). Valid pixel coordinates run `0 .. w-1` and `0 .. h-1` --
the convention `get_laf_center` documents and the one `normalize_laf`/`denormalize_laf` already use -- so the
upper bound is now `w - 1 - border` and `h - 1 - border`. A LAF whose boundary points reach past the last valid
pixel coordinate is now reported as outside; anything strictly inside is unchanged. The equivalent inlined check
in `ScaleSpaceDetector._process_octave` moved with it, so detections within one pixel of the right or bottom edge
that used to survive its `border=5` filter are now discarded. That inlined check also computed its `max |sin|`
constant with the wrong angular spacing (`2*pi/11` instead of the `2*pi/10` that `laf_to_boundary_points(n_pts=12)`
actually samples), inflating the tested x-extent by 4% and making the inline check stricter than the reference it
claims to reproduce; the two now agree exactly. That correction is a loosening in x, so it can change
`ScaleSpaceDetector` output on its own: a detection whose x-extent falls between the two constants used to be
discarded and is now kept. It is rare -- the detection has to land in a band of width `0.039 * half_s` against the
x bound -- but when it fires it can promote the strongest response in the image, so it is not output-neutral.
