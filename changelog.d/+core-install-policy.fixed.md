`kornia.core.external.LazyLoader` can be copied, deep-copied and pickled (these recursed or failed on the module
object), and dunder lookups such as `__wrapped__` no longer import, install or ask for a module that is not loaded yet.
An installed optional module now loads when `--doctest-modules` is in `sys.argv`: the `onnxruntime` and `diffusers`
loaders used to stay empty in any such process, for example a user's own `pytest --doctest-modules` run, and raised
`AttributeError: 'NoneType' object has no attribute ...`. `ImageModule` and the augmentation containers no longer
import PIL through the lazy loader to check an argument that is not an image (a keyword such as `scale=0.5` or
`data_keys=[...]`), so those calls work on an install without the `image` extra.
