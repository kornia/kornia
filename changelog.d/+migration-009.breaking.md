`nms2d(x, (1, 1))` and `nms3d(x, (1, 1, 1))` return `x` unchanged. A unit window has no neighbours,
so nothing can be suppressed; the old 2-D result was an artifact of the zeroed centre tap of the
convolution kernel summing to `0.0`, while the old 3-D path raised because of #4241. (#4242)
