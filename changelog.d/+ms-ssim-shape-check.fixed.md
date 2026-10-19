`MS_SSIMLoss` now rejects mismatched input and target shapes with a clear error instead of silently
broadcasting them (for example a batch-2 input against a batch-1 target) and returning a wrong loss,
and rejects inputs that are not 4-D `BxCxHxW`.
