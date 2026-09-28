`AUTUMN` is dropped from `kornia.color.__all__` (#4143). The class was deprecated in 0.7.2 in favour of
`ColorMap(base='autumn')` and removed by #3432, which left the `__all__` entry behind. Since that removal
`from kornia.color import AUTUMN` has raised `AttributeError` and `from kornia.color import *` has failed on
it, so nothing that works today stops working — the entry named a binding that no longer existed. Callers
still reaching for the class should use `ColorMap(base='autumn')`.
