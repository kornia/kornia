# Kornia Repository Instructions

This file provides instructions for GitHub Copilot when working with code in this repository.

## Coding Standards

Follow the coding standards and best practices defined in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards).

## AI-Generated Content Policy

- Code and comments must not be direct, unreviewed outputs of AI agents
- All AI-assisted contributions require human oversight and validation
- Ensure code logic reflects genuine understanding, not copied AI output

## Key Guidelines

- **Code style**: Follow PEP8, use 120 character line length, Ruff linting, and f-strings
- **Type hints**: Required for all function inputs and outputs
- **Documentation**: Use Google docstring convention
- **Testing**: Use `BaseTester` pattern with smoke, exception, cardinality, feature, gradcheck, and dynamo tests
- **Dependencies**: Only PyTorch is allowed as a dependency

## Running Checks

```bash
pixi run lint       # Linting
pixi run typecheck  # Type checking
pixi run test       # Testing
pixi run doctest    # Documentation tests
```

## Review Checklist

When reviewing code changes, verify:

- Code and comments are not direct, unreviewed AI agent outputs
- Code follows guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md)
- Tests are included for new functionality
- Code passes `pixi run lint` and `pixi run typecheck`
