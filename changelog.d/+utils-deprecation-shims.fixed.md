The deprecated `kornia.utils` names and the deprecated top-level aliases such as `kornia.create_meshgrid` work as the
0.8.2 functions did again. `kornia.utils.<name>` resolves after a plain `import kornia`, where it raised
`AttributeError` unless `kornia.utils` had been imported first, and the 0.8.2 path
`kornia.utils.sample.get_sample_images` is back as a deprecated alias. Each shim takes the signature of the function it
forwards to, so `inspect.signature` and `help()` show the real parameters, and code that calls one compiles with
`torch.compile(fullgraph=True)` and scripts with `torch.jit.script` again: the deprecation warning is skipped while a
graph is captured and inside TorchScript, and eager calls still warn. `kornia.losses.one_hot` scripts with
`torch.jit.script`. `torch.load(..., map_location=kornia.utils.map_location_to_cpu)` warns once per call, at the line
that called `torch.load`, so the default warning filters show it. `from kornia import to_jax`, and the same for
`to_numpy`, `to_tensorflow` and `kornia.transpiler`, imports again, and calling the name raises the `RuntimeError` that
explains the removal instead of a bare `ImportError`. The deprecation messages no longer say that the `kornia.utils`
module has been removed.
