# Copilot instructions for kornia

Use [AGENTS.md](../AGENTS.md) as the repository's operational guide. In particular:

- prefer existing kornia utilities and nearby code patterns;
- preserve device, dtype, batching, gradient, JIT, and `torch.compile` behavior where applicable;
- use `BaseTester`, injected `device`/`dtype` fixtures, and `self.assert_close()` in tests;
- treat half precision, CUDA, MPS, and ONNX as explicit compatibility surfaces;
- run focused checks and report what was and was not verified;
- avoid requesting or exposing private environment information.

When reviewing, comment on concrete correctness, compatibility, maintainability, documentation, or test issues and tie each suggestion to the code under review.
