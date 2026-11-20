`spatial_gradient3d` scales its first-order difference buffer in place, avoiding a second output-sized allocation while preserving input values and gradients.
