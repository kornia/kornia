The mutual-information losses (`mutual_information_loss_2d`, `mutual_information_loss_3d`, their normalized variants
and the `FromRef` modules) now accept empty batches instead of raising a reshape error, and a signal with no samples
raises a clear error instead of an `IndexError`.
