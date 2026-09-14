`MultiResolutionDetector.detect` now returns the `num_features` highest responses in the image (#4222). Each
pyramid level used to be asked only for its own share of the budget, so a level could contribute at most
`quotas[0]` (the upscaled level) or `sum(quotas[: i + 1 + up_levels])` (pyramid level `i`) -- both well under
`num_features` for the finer levels. When an image's strongest maxima concentrated on one level, which is the
common case rather than a corner case, the excess was never requested and so could not be ranked in by the
final global `topk`: the detector silently returned something other than the top-k, and the sorted responses of
`detect(k)` were not a prefix of those of `detect(k')`. Every level is now searched for `num_features` candidates, which is exactly sufficient
because the detections a level contributes to the global top-k are that level's own strongest. The per-level
apportionment, including #4101's largest-remainder rounding, is removed with it. This changes which keypoints
are returned for a given image: callers get strictly better-ranked detections, and any result that depended on
the old per-level spread will differ. The prefix is on the sorted responses, not on which detection carries a
given response: `topk` breaks a tie by position, so exactly equal maxima can come back in a different order for
a different `num_features`. It carries about 1.2x the candidates through the concatenation, which is not
measurable next to the response function and the non-maxima suppression.
