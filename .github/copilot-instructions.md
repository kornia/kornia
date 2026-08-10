# Kornia Repository Instructions

This file provides instructions for GitHub Copilot when working with code in this repository.

## Coding Standards

Follow the coding standards and best practices defined in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards).

## AI Policy

**CRITICAL**: All contributions must comply with the [AI_POLICY.md](../AI_POLICY.md). Review that document for complete requirements.

### Core Principles:
- The policy gates on **verification and accountability**, not on how much AI was used
- Every functional change must carry pasted proof of local execution (test logs)
- The submitter must be able to explain every line; unexplainable code fails review
- AI-generated PRs are acceptable when honestly disclosed, verified, and explainable; mislabeled disclosure or imaginary verification is grounds for closure
- Hallucinated or redundant comments are the fastest tell of an unreviewed PR — flag them

## Instructions for AI Reviewers (Copilot / CodeRabbit)

AI-based reviewers (e.g. GitHub Copilot, CodeRabbit) must follow the repository's AI usage policy and review rules.

For the complete and authoritative AI reviewer instructions, see [AI_POLICY.md](../AI_POLICY.md), section 3.

When generating or reviewing suggestions, prefer:
- Enforcing the coding standards in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards)
- Enforcing the AI usage rules and review heuristics defined in [AI_POLICY.md](../AI_POLICY.md)
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

- Code follows guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md)
- Code complies with [AI_POLICY.md](../AI_POLICY.md)
- Tests are included for new functionality
- Code passes `pixi run lint` and `pixi run typecheck`
- PR includes proof of local test execution (test logs)
- Code uses `kornia` utilities instead of reinventing existing functionality
- Comments are written in English, are non-redundant, and reflect genuine understanding of the code
- The AI Usage Disclosure section is completed; if code quality signals (hallucinated comments, reinvented utilities) contradict the disclosure, note the mismatch

## Lane-Aware PR Review

Kornia uses a two-lane contribution model (see
[CONTRIBUTING.md](../CONTRIBUTING.md#which-lane-is-your-contribution)). Determine the
PR's lane from its template declaration and content, then apply the matching checks.

1. **Green lane** (test-first bug fixes, verified docs fixes, `help wanted` issues,
   benchmark results) — no issue link or assignment is required. Verify instead:
   - Bug fixes: the PR contains a new test and states that it fails on `main` without
     the fix; test logs are pasted
   - Docs fixes: the PR description shows the verification snippet that was run
   - `help wanted` PRs: the referenced issue carries the `help wanted` label and the
     PR meets the acceptance criteria stated in that issue; if another open PR
     already addresses the same issue, note it
   - Benchmark contributions: results follow the `benchmarks/` `--contribute` format
   - The PR does NOT introduce new features, new public APIs, or behavior changes —
     if it does, it is not green lane; ask for the discuss-first process

2. **Discuss-first lane** (features, API changes, behavior/default/convention changes,
   models, large refactors):
   - Verify the PR description contains a valid issue reference (e.g., "Fixes #123")
   - Verify a maintainer confirmed the scope on the linked issue (a confirming
     comment or label; formal assignment is optional)
   - **Scope matching (critical)**: the PR implementation strictly matches what the
     issue describes; changes beyond that scope should be split into separate
     issues/PRs
   - **Behavior/default/convention changes deserve maximum scrutiny**: in a geometry
     library, changed semantics break users silently. Confirm the issue discussion
     explicitly acknowledges the behavior change and its migration path

**Reviewer Action**: If requirements for the PR's lane are not met, explain exactly
what is missing (which evidence for green lane; which confirmation for discuss-first)
rather than issuing a generic warning. Never recommend closure based on subjective
quality opinions alone — anchor findings in the objective criteria above.
