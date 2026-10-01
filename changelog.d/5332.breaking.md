Reject fewer than two samples with `ValueError` in unbiased ZCA whitening instead of producing NaNs or an empty result; biased whitening (`unbiased=False`) still supports a single sample.
