`kornia.enhance.solarize` accepts an `int` `thresholds` or `additions`, as its docstring says, and reads it as the equal `float`. It used to raise `TypeError`.
