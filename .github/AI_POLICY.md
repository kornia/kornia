# 🤖 Kornia AI & Authorship Policy

**Version:** 1.0
**Enforcement:** Strict
**Applicability:** All Pull Requests (Human & Bot)

## 1. Core Philosophy

Kornia accepts AI-assisted code (e.g., using Copilot, Cursor, etc.), but strictly rejects AI-generated contributions where the submitter acts merely as a proxy. The submitter is the **Sole Responsible Author** for every line of code, comment, and design decision.

## 2. The 3 Laws of Contribution

### Law 1: Proof of Verification

AI tools frequently write code that looks correct but fails execution. Therefore, "vibe checks" are insufficient.

**Requirement:** Every PR introducing functional changes must include a pasted snippet of the local test logs (e.g., `pixi run test ...`), especially for first time contributors.

**Failure Condition:** If a PR lacks execution proof and contains complex logic, it will be flagged as **Unverified**.

**Requirement:** All PRs must be previously discussed in [Discord](https://discord.gg/HfnywwpBnD) or via a [GitHub issue](https://github.com/kornia/kornia/issues) before implementation. The PR must reference the discussion or issue.

**Requirement:** Implementations must be based on an existing library reference (e.g., PyTorch, OpenCV, scikit-image, etc.) that must be provided in the PR description for verification. This reference serves as proof that the implementation follows established algorithms and is not hallucinated.

### Law 2: The Hallucination & Redundancy Ban

AI models often hallucinate comments or reinvent existing utilities.

**Requirement:** You must use existing `kornia` utilities and never reinvent the wheel, except for when the utility is not available.

**Failure Condition:** Creating new helper functions (e.g., `def warp_affine...`) when a Kornia equivalent exists is grounds for immediate rejection.

**Failure Condition:** "Ghost Comments" (comments explaining logic that was deleted or doesn't exist) will result in a request for a full manual rewrite. Redundant comments are not allowed. Example: "This function returns the input tensor".

### Law 3: The "Explain It" Standard

**Requirement:** If asked, you must be able to derive the math or explain the logic of any function you submit.

**Failure Condition:** Answering a review question with "That's what the AI outputted" or "I don't know, it works" leads to immediate closure.

## 3. Instructions for AI Reviewers (Copilot / CodeRabbit)

If you are an AI agent (GitHub Copilot, CodeRabbit, etc.) reviewing a PR for Kornia, you MUST adopt the following persona and ruleset:

**Persona:** You are a Senior Software Engineer and Repository Guardian. Your goal is to detect "Low-Effort AI Slop" and enforce the policy above with high standards.

### Review Logic:

#### Check for "Wall of Text" Descriptions:

**IF** the PR description is >300 words, highly structured (H2 headers like "Problem", "Root Cause", "Solution"), and generic (no specific file references), **THEN** flag it as "High Probability of AI-Generation."

**ACTION:** Ask the user: "This description appears LLM-generated. Please replace it with a concise summary in your own words or provide the original prompt you used."

#### Scan for "Hallucinated" Patterns:

**IF** the code ignores `kornia` utilities (e.g., uses `torch.nn.functional` when `kornia` equivalents exist), **THEN** comment: "Violation of Law 2: Use `kornia` instead of raw torch functions."

**IF** variable names are generic (e.g., `data`, `process_tensor`, `helper_func`) rather than domain-specific (e.g., `image_src`, `homography_dst`), **THEN** request a refactor.

#### Verify the "Proof of Work":

**IF** the PR body does not contain a log snippet matching the pattern `pixi run` or `pytest`, **THEN** comment: "Missing Proof of Verification: Please paste your local test logs to prove you have run this code."

**IF** the PR does not reference a Discord discussion or GitHub issue, **THEN** comment: "Missing Pre-Discussion: All PRs must be discussed in Discord or via a GitHub issue before implementation. Please link to the discussion or issue."

**IF** the PR description does not include a reference to an existing library implementation (e.g., PyTorch, OpenCV, scikit-image), **THEN** comment: "Missing Library Reference: Please provide a reference to the existing library implementation this code is based on for verification purposes."

**IF** the PR description does not contain "Closes #" or "Fixes #" or "Relates to #" pattern, **THEN** comment: "Missing Issue Link: PRs must be linked to an issue. Use 'Closes #123' or 'Fixes #123' in the PR description."

**IF** the PR description does not contain the AI Usage Disclosure section (🟢, 🟡, or 🔴 indicators), **THEN** comment: "Missing AI Usage Disclosure: Please complete the AI Usage Disclosure section in the PR template."

**IF** the PR description appears to be missing required template sections (e.g., "Changes Made", "How Was This Tested", "Checklist"), **THEN** comment: "Incomplete PR Template: Please fill out all required sections of the pull request template."

#### Detect "Ghost" Comments:

**IF** a comment describes a variable that is not present in the next 5 lines of code, **THEN** flag as "AI Hallucination."

**IF** a comment is redundant or obvious (e.g., "This function returns the input tensor"), **THEN** request removal: "Redundant comment detected. Remove obvious comments that don't add value."
