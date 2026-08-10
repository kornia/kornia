# Kornia Repository Instructions

This file provides instructions for GitHub Copilot when working with code in this repository.

## Coding Standards

Follow the coding standards and best practices defined in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards).

## AI Policy

**CRITICAL**: All contributions must comply with the [AI_POLICY.md](../AI_POLICY.md). Review that document for complete requirements.

### Core Principles:
- The policy gates on **verification and accountability**, not on how much AI was used
- Every functional change carries pasted test logs as execution evidence — evidence is ranked per AI_POLICY.md Rule 1 (oracle/invariant-backed test > failing-test repro > CI > local logs); logs alone are provenance, not proof
- The submitter owns the code and must be able to address targeted questions about it (algorithms, shapes, edge cases, design); the failure condition is an unresolved correctness or ownership concern, never eloquence or language fluency
- AI-generated PRs are acceptable when honestly disclosed and verified; only **demonstrable deception** (verification that never ran, fabricated logs) is grounds for closure — never an arguable 🟡/🔴 classification
- Hallucinated or redundant comments are a code-quality defect — flag them as such

## Instructions for AI Reviewers (Copilot / CodeRabbit)

AI-based reviewers (e.g. GitHub Copilot, CodeRabbit) must follow the repository's AI usage policy and review rules.

For the governing contribution policy, see [AI_POLICY.md](../AI_POLICY.md). This file contains the canonical operational instructions for AI reviewers.

When generating or reviewing suggestions, prefer:
- Enforcing the coding standards in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards)
- Enforcing the AI usage rules and review heuristics defined in [AI_POLICY.md](../AI_POLICY.md)
- Highlighting missing tests, missing execution evidence (test logs, oracle-backed expected values), and misuse of `kornia` vs. raw PyTorch utilities
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
- PR includes pasted test logs as execution evidence
- Code uses `kornia` utilities instead of reinventing existing functionality
- Comments are written in English, are non-redundant, and reflect genuine understanding of the code
- The AI Usage Disclosure section is completed. **Never infer AI use from code or writing style** — poor code is not evidence of AI use, and disclosure classification is not litigated (see AI_POLICY.md Rule 4). Flag quality problems (hallucinated comments, reinvented utilities) as quality problems in their own right, not as disclosure evidence

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
