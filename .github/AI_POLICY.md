# 🤖 Kornia AI & Authorship Policy

**Version:** 1.0
**Enforcement:** Strict
**Applicability:** All Pull Requests (Human & Bot)

## 1. Core Philosophy

Kornia accepts AI-assisted code (e.g., using Copilot, Cursor, etc.), but strictly rejects AI-generated contributions where the submitter acts merely as a proxy. The submitter is the **Sole Responsible Author** for every line of code, comment, and design decision.

## 2. The 3 Laws of Contribution

### Law 1: Proof of Verification

AI tools frequently write code that looks correct but fails execution. Therefore, "vibe checks" are insufficient.

**Requirement:** Every PR introducing functional changes must include a pasted snippet of the local test logs (e.g., `pixi run test ...`). This is mandatory for all contributors and is particularly important for first-time contributors.

**Failure Condition:** If a PR lacks execution proof and contains complex logic, it will be flagged as **Unverified**.

**Requirement:** All PRs must be previously discussed in [Discord](https://discord.gg/HfnywwpBnD) or via a [GitHub issue](https://github.com/kornia/kornia/issues) before implementation. The PR must reference the discussion or issue.

**Requirement:** Implementations must be based on an existing library reference (e.g., PyTorch, OpenCV, scikit-image, etc.) that must be provided in the PR description for verification. This reference serves as proof that the implementation follows established algorithms and is not hallucinated.

### Law 2: The Hallucination & Redundancy Ban

AI models often hallucinate comments or reinvent existing utilities.

**Requirement:** You must use existing `kornia` utilities and never reinvent the wheel, except for when the utility is not available.

**Failure Condition:** Creating new helper functions (e.g., `def warp_affine...`) when a Kornia equivalent exists is grounds for immediate rejection.

**Failure Condition:** "Ghost Comments" (comments explaining logic that was deleted or doesn't exist) will result in a request for a full manual rewrite. Redundant comments are not allowed. Example: "This function returns the input tensor".

### Law 3: The "Explain It" Standard

**Requirement:** If a maintainer or reviewer asks during code review, you must be able to derive the math or explain the logic of any function you submit.

**Failure Condition:** Answering a review question with "That's what the AI outputted" or "I don't know, it works" leads to immediate closure.

## 3. Instructions for AI Reviewers (Copilot / CodeRabbit)

If you are an AI agent (GitHub Copilot, CodeRabbit, etc.) reviewing a PR for Kornia, you must follow the repository’s dedicated reviewer instructions.

The **canonical and up-to-date instructions for AI reviewers** are maintained in [`.github/copilot-instructions.md`](./copilot-instructions.md). That document defines:

- The expected reviewer persona and responsibilities
- The checks to perform on PR descriptions, code, tests, and comments
- The required enforcement of the laws defined in this `AI_POLICY.md`

Any other document (including this one) should treat `copilot-instructions.md` as the single source of truth for AI reviewer behaviour. When updating reviewer logic, update `copilot-instructions.md` first and, if needed, adjust references here.

This section exists to link AI reviewers to the canonical instructions and to make clear that those instructions must enforce the policies defined in Sections 1 and 2 above.

## 4. Additional Resources

For comprehensive guidance on contributing to Kornia, including development workflows, code quality standards, testing practices, and AI-assisted development best practices, see the [Best Practices section](../CONTRIBUTING.md#best-practices) in `CONTRIBUTING.md`.
