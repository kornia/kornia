`histogram2d` accumulates half-precision joint counts and normalization in float32 to avoid zero or NaN histograms on ordinary image sizes, while preserving the output dtype.
