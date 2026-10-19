Fix `match_fginn`'s geometric consistency check comparing every query's candidates against **query 0**'s
candidates instead of against the query's own 1st nearest neighbor, which changes every `match_fginn` and
`GeometryAwareDescriptorMatcher("fginn")` result -- the latter being that class's *default* mode (#4062). The distance
`kdist[i, k] = || xy2[idx[i, k]] - xy2[idx[0, k]] ||` was measured from a slice of dim 0 that broadcast query 0's
whole candidate list over the batch; it is now `candidates_xy[:, 0:1]`, i.e.
`kdist[i, k] = || xy2[idx[i, k]] - xy2[idx[i, 0]] ||`. Three consequences disappear with it. The geometric term
was inert -- an unrelated query's candidates are almost always farther apart than `spatial_th`, so nothing was
penalized and the raw 2nd nearest neighbor was used, making `match_fginn` behave as `match_snn`: on a 300x300
descriptor set with planted near-duplicates, 15 of the 15 matches shared with `match_snn` had an identical ratio
before and 14 of 15 after. Query 0 always matched, since `candidates_xy[0] - candidates_xy[0]` is identically zero,
so all of its candidates were penalized and its ratio collapsed to ~0. And a query's ratio depended on the other
queries in the batch: with only query 0 changed, one query's ratio moved between `0.9005` and `2.8e-08`. The seven
existing FGINN tests pass unchanged on both sides, which is how this survived; two tests that discriminate it were
added. `match_fginn`'s docstring now also records the saturation corner, where every candidate falls within
`spatial_th` of the 1st nearest neighbor, the ratio collapses towards zero and the match is accepted.
