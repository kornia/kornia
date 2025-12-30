# Python Code Review Guidelines for Kornia

When reviewing Python code in this project, follow the coding standards and best practices defined in **[CONTRIBUTING.md](../../CONTRIBUTING.md)**.

## Key Review Points

### AI-Generated Content Policy

- Code and comments must not be direct, unreviewed outputs of AI agents
- All AI-assisted contributions require human oversight and validation
- Ensure code logic reflects genuine understanding, not copied AI output

### Quick Reference

For detailed guidelines on the following topics, see the [Coding Standards](../../CONTRIBUTING.md#coding-standards) section:

- **Code style**: PEP 8, 120 char line length, Ruff linting, f-strings
- **Type hints**: Required for all function inputs and outputs
- **Documentation**: Google docstring convention
- **Testing**: `BaseTester` pattern with smoke, exception, cardinality, feature, gradcheck, and dynamo tests
- **Dependencies**: Only PyTorch allowed

### Running Checks

```bash
pixi run lint       # Linting
pixi run typecheck  # Type checking
pixi run test       # Testing
pixi run doctest    # Documentation tests
```

## Review Checklist

- [ ] Code and comments are not direct, unreviewed AI agent outputs
- [ ] Code follows guidelines in [CONTRIBUTING.md](../../CONTRIBUTING.md)
- [ ] Tests are included for new functionality
- [ ] Code passes `pixi run lint` and `pixi run typecheck`
