Non-maxima suppression applies one border rule at every window size. `NonMaximaSuppression2d` /
`nms2d` with a window larger than `(7, 7)`, and `NonMaximaSuppression3d` / `nms3d`, no longer report
maxima inside the `(k - 1) // 2` border strip. Previously the general path replicate-padded its input,
so a position whose window ran off an edge was judged against duplicated copies of the edge values,
which both fabricated plateaus that suppressed genuine maxima and hid genuine neighbours that should
have suppressed spurious ones; the `(3, 3)`, `(5, 5)` and `(7, 7)` paths meanwhile rejected the strip
outright, so the two disagreed about the same pixel. The explicit paths' rule is now the rule
everywhere. Every kornia detector already rejects that strip -- ALIKED by hand after calling `nms2d`,
DISK and XFeat by using `k = 5`, `MultiResolutionDetector` by zeroing 15 px before the call -- so no
detector output changes. (#4239, #4242)
