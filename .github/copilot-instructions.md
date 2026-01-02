# Kornia Repository Instructions

This file provides instructions for GitHub Copilot when working with code in this repository.

## Coding Standards

Follow the coding standards and best practices defined in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards).

## AI Policy

**CRITICAL**: All contributions must comply with the [AI_POLICY.md](AI_POLICY.md). Review that document for complete requirements.

### Core Principles:
- Code and comments must not be direct, unreviewed outputs of AI agents
- All AI-assisted contributions require human oversight and validation
- Ensure code logic reflects genuine understanding, not copied AI output
- The submitter is the Sole Responsible Author for every line of code

## Instructions for AI Reviewers (Copilot / CodeRabbit)

AI-based reviewers (e.g. GitHub Copilot, CodeRabbit) must follow the repository's AI usage policy and review rules.

For the complete and authoritative AI reviewer instructions, see [AI_POLICY.md](AI_POLICY.md), section 3.

When generating or reviewing suggestions, prefer:
- Enforcing the coding standards in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards)
- Enforcing the AI usage rules and review heuristics defined in [AI_POLICY.md](AI_POLICY.md)
- Highlighting missing tests, missing proof of local execution, and misuse of `kornia` vs. raw PyTorch utilities
## Key Guidelines

- **Code style**: Follow PEP8, use 120 character line length, Ruff linting, and f-strings
- **Type hints**: Required for all function inputs and outputs
- **Documentation**: Follow documentation and docstring guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards) and match the existing codebase style
- **Testing**: Use `BaseTester` pattern with smoke, exception, cardinality, feature, gradcheck, and dynamo tests
- **Dependencies**: Only PyTorch is allowed as a dependency
- **Use kornia**: Always prefer `kornia` utilities over raw PyTorch functions

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
- Code complies with [AI_POLICY.md](AI_POLICY.md)
- Tests are included for new functionality
- Code passes `pixi run lint` and `pixi run typecheck`
- PR includes proof of local test execution (test logs)
- Code uses `kornia` utilities instead of reinventing existing functionality
- Comments are written in English and verified by a human with a good understanding of the code
