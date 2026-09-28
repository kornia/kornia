# Claude guide for kornia

@AGENTS.md

The operational guide is [AGENTS.md](AGENTS.md); the line above imports it for Claude Code.

When helping a user in this repository:

- Start from the smallest relevant code path and run focused checks before making broad claims.
- Prefer existing kornia utilities and nearby patterns over introducing a parallel abstraction.
- Preserve tensor device, dtype, batch semantics, gradients, and `torch.compile` behavior when they apply.
- Treat float16, bfloat16, MPS, CUDA, JIT, and ONNX as distinct compatibility surfaces; do not assume a CPU float32 result covers them.
- Explain what you verified and what remains unverified.
- Ask before gathering broad environment information. Request only useful versions and hardware details, and remind users to redact paths, usernames, tokens, hostnames, and private data names.

If you help prepare a contribution, summarize what you changed and checked so the user can reuse that information in the pull request.
