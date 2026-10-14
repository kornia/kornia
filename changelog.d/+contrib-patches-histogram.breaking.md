`histogram_matching` with an empty template now raises a `ValueError` ("histogram_matching needs a non-empty template."); it used to raise an `IndexError` from inside the quantile computation.
