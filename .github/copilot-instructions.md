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

## PR-Issue Alignment Review

When reviewing pull requests, ensure strict alignment with the linked issue:

1. **Issue Link Verification**:
   - Verify the PR description contains a valid issue reference (e.g., "Fixes #123" or "Closes #123")
   - Confirm the linked issue exists and is open (or was open when the PR was created)

2. **Assignment Verification**:
   - Check that the PR author is assigned to the linked issue
   - If not assigned, request that a maintainer assign the issue before proceeding with review

3. **Scope Matching**:
   - **Critical**: Verify that the PR implementation strictly matches what the issue describes
   - The PR should not include changes beyond the scope of the linked issue
   - If the PR includes additional features or changes not mentioned in the issue, request that those be split into separate issues/PRs
   - Compare the PR description, code changes, and tests against the issue description to ensure alignment

4. **Issue Approval Status**:
   - Verify the linked issue has been reviewed and approved by a maintainer
   - Issues with the `triage` label may not have been fully reviewed yet

**Reviewer Action**: If the PR does not match the issue scope or requirements, clearly explain the mismatch and request that the PR be updated to strictly align with the issue, or that additional changes be moved to separate issues/PRs.
