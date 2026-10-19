`RandomTransplantation` and `RandomTransplantation3D` transplanted nothing on MPS when no `excluded_labels`
were given: PyTorch's MPS backend evaluates `all()` over the empty excluded-label axis to an undefined value,
usually `False`, so every donor label was filtered out and the output equalled the input. The filter is now skipped when there is
nothing to exclude. (#4160)
