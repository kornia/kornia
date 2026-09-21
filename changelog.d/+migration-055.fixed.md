`jpeg_codec_differentiable` no longer returns an all-`NaN` image at `jpeg_quality=0`, a value inside the
`[0, 100]` range it documents and validates. `_jpeg_quality_to_scale` divides `5000` by the quality on the
`< 50` branch, which is `inf` at `0`, and the polynomial floor of `inf` is `NaN`, poisoning the image and
its gradient. A quality of exactly `0` is now given the scale of quality `1`, which is the table libjpeg
gives quality `0`. The guard is that one point: a fractional quality in `(0, 1)` was already finite, keeps
its own larger scale, and is untouched, so this endpoint is not the limit of the formula from above.
Output and gradients for every quality `> 0` are byte-identical. The tests drew
the quality with `torch.randint(low=0, high=100)` unseeded at sixteen sites, so roughly one run in twenty of
`tests/enhance/test_jpeg.py` went red on any job; the draws now start at `1` and the boundary has pinned
cases of its own. (#4205)
