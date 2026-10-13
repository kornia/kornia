Removed `kornia.to_tensorflow()`, `kornia.to_jax()` and `kornia.to_numpy()`, along with the
multi-framework-support advertising in the README and the docs landing page. These functions used
to lazily transpile the library to TensorFlow, JAX or NumPy through the third-party `ivy` package
(an optional `dev`/`docs` dependency, now also removed). Testing in September 2026 found the
integration unreliable across all three targets — a checkout-path crash in Ivy's module walker, a
torch/triton segfault when `transformers` is also installed, an undocumented `flax` requirement for
JAX, and a `jaxlib`/`numpy` API-compatibility break — against Ivy's latest release (1.0.0.5, June
2025), with the upstream project showing little ongoing maintenance. The three names still
import, at the top level and from `kornia.transpiler`, so `from kornia import to_jax` and
`hasattr(kornia, "to_jax")` keep working; calling any of them raises a `RuntimeError` that says
what was removed and links the page below. See `get-started/multi-framework-support` for what
was tested and why. (#4196)
