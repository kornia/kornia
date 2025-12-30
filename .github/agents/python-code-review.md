# Python Code Review Guidelines for Kornia

This document provides guidelines for reviewing Python code in the Kornia project. Follow these best practices to ensure consistency and quality across the codebase.

## AI-Generated Content Policy

- **Human oversight required**: Code and comments must not be direct, unreviewed outputs of AI agents
- AI-generated PRs without human oversight will be flagged and may be closed
- All AI-assisted contributions must be reviewed and validated by a human before submission
- Ensure code logic is understood and verified, not just copied from AI output
- Comments should reflect genuine understanding, not generic AI-generated explanations

## Code Style

### Formatting and Linting
- **Line length**: Maximum 120 characters
- **Linter**: Use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- **Style guide**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Import ordering**: Use isort conventions with `kornia` as first-party
- **Strings**: Use f-strings for string formatting ([PEP 498](https://peps.python.org/pep-0498/))

### License Header
All Python files must include the Apache 2.0 license header at the top:
```python
# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
```

## Type Hints

### Requirements
- **Always** type function inputs and outputs
- Use Python 3.11+ typing features
- Import `Tensor` from `kornia.core`: `from kornia.core import Tensor`
- For non-JIT modules, use `from __future__ import annotations`

### Example
```python
from __future__ import annotations
from kornia.core import Tensor

def homography_warp(
    patch_src: Tensor,
    dst_homo_src: Tensor,
    dsize: tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros'
) -> Tensor:
    ...
```

### JIT Compatibility
- Use `Optional[]`, `Union[]` instead of `|` for torch.jit scripting
- Use `List[]` instead of `list` for torch.jit scripting
- Do not use `from __future__ import annotations` in JIT modules

## Documentation

### Docstrings
- Use [Google docstring convention](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Document all public functions, classes, and methods
- Include `Args`, `Returns`, `Raises`, and `Example` sections where appropriate

### Example
```python
def my_function(x: Tensor, threshold: float = 0.5) -> Tensor:
    """Short description of the function.

    Longer description if needed, explaining the behavior
    and any important details.

    Args:
        x: Input tensor of shape :math:`(B, C, H, W)`.
        threshold: Threshold value for processing. Default: 0.5.

    Returns:
        Output tensor of shape :math:`(B, C, H, W)`.

    Raises:
        ValueError: If threshold is not in range [0, 1].

    Example:
        >>> x = torch.rand(1, 3, 4, 4)
        >>> result = my_function(x, threshold=0.3)
    """
```

## Testing

### Test Structure
- Use the `BaseTester` class from `testing.base`
- Tests should cover: smoke tests, exceptions, cardinality, features, gradcheck, and dynamo

### Test Pattern
```python
from testing.base import BaseTester

class TestMyFunction(BaseTester):
    def test_smoke(self, device, dtype):
        # Test basic functionality with various parameters
        pass

    def test_exception(self, device, dtype):
        # Test expected exceptions
        pass

    def test_cardinality(self, device, dtype):
        # Test output shapes
        pass

    def test_feature_foo(self, device, dtype):
        # Test specific feature
        pass

    def test_gradcheck(self, device):
        # Test gradients using self.gradcheck(...)
        pass

    def test_dynamo(self, device, dtype, torch_optimizer):
        # Test with dynamo optimizer
        pass
```

### Test Coverage
- Cover different devices (`cpu`, `cuda`, `mps`)
- Cover different dtypes (`float32`, `float64`, `float16`, `bfloat16`)
- Cover different batch sizes

## Dependencies

### Third-Party Libraries
- **Not allowed**: Only PyTorch is permitted as a dependency
- Use existing kornia utilities instead of external libraries

## Code Quality Checks

### Pre-commit Hooks
The project uses pre-commit hooks for:
- End-of-file fixer
- Trailing whitespace removal
- YAML/TOML validation
- Merge conflict detection
- Large file detection
- Ruff linting and formatting
- Codespell for typo detection

### Type Checking
- Run type checking with `pixi run typecheck` (uses `ty`)
- Ensure no type errors are introduced

### Running Checks Locally
```bash
# Linting
pixi run lint

# Type checking
pixi run typecheck

# Testing
pixi run test
pixi run test tests/<specific_test>.py

# Doctests
pixi run doctest
```

## Review Checklist

When reviewing Python code, verify:

- [ ] Code and comments are not direct, unreviewed AI agent outputs
- [ ] Code follows PEP 8 style guidelines
- [ ] Line length does not exceed 120 characters
- [ ] License header is present on new files
- [ ] Type hints are provided for all function inputs and outputs
- [ ] Docstrings follow Google convention
- [ ] Tests are included for new functionality
- [ ] Tests cover smoke, exception, cardinality, and gradcheck scenarios
- [ ] No third-party library dependencies are added
- [ ] Code passes `pixi run lint`
- [ ] Code passes `pixi run typecheck`
- [ ] All existing tests still pass
