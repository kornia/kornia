Fix `ZCAWhitening.inverse_transform` to use the fitted sample dimension, restoring data correctly for nonzero and negative `dim` values instead of raising shape errors or mixing feature values.
