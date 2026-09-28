Optional-dependency extras `kornia[onnx]` and `kornia[sd]` declare the third-party packages that
the ONNX and Stable-Diffusion-dissolving wrappers lazily import, and are documented on the
installation page. Missing-dependency errors now name the extra to install, and `dev` no longer
pulls `diffusers` and `transformers`. (#4301)
