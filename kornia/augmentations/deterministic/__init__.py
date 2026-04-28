"""kornia.augmentations.deterministic — zero-randomness, export-clean transforms.

Transforms in this namespace:
- Have no randomness (no ``random_apply``, no per-sample params)
- Are ``torch.export``-clean (verified in PR-EX)
- May or may not preserve gradients (default: no_grad; ``NormalizeWithGrad`` is the
  one carved-out exception)
"""
from kornia.augmentations.deterministic.normalize_with_grad import NormalizeWithGrad

__all__ = ["NormalizeWithGrad"]
