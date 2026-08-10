# 🤖 Kornia AI & Authorship Policy

**Version:** 2.0
**Applies to:** every pull request, human or bot.

## Why this policy exists — the honest version

AI tools can now produce code that looks right in seconds. Some of it is genuinely
good. A lot of it compiles, reads confidently, and is quietly wrong — and reviewing it
costs a volunteer maintainer far more time than generating it cost the submitter.
Around application deadlines (GSoC and similar) we have received whole waves of these.

At the same time, let's be honest about ourselves: **Kornia maintainers use AI tools
heavily** — including entire agent-written pull requests. Banning "AI-generated code"
would be hypocritical, and it would also miss the point. What we have learned by
working this way is that the difference between an excellent AI-built PR and a useless
one is never the tool. It is whether every claim was **executed and verified**, and
whether a **human stands behind the result** and can explain it.

So this policy does not gate on how much AI you used. It gates on two things:

1. **Verification** — did you actually run it, and can you show us?
2. **Accountability** — do you understand it, and will you answer for it?

Unverified code is a problem. Unexplainable code is a problem. AI is neither.

## The rules

### Rule 1 — Show us the evidence

Code that "looks correct" is not enough, no matter who or what wrote it. And one
honest caveat about our own favorite kind of evidence: **"fails on `main`, passes
with the fix" proves the behavior *changed* — not that the new behavior is
*correct*.** A generated test can encode the same wrong assumption as generated
code, and a pasted log can in principle be fabricated. So evidence is ranked,
strongest first:

1. **A test backed by a credible oracle** — a reference implementation (OpenCV,
   scipy, a paper — embedded as a hardcoded literal with its generation snippet), a
   documented contract, or a mathematical invariant (`warp(H⁻¹) ∘ warp(H) ≈ id`).
   For numerical and geometry claims this is the bar.
2. **A failing-test-first repro** — fails on `main`, passes with the fix. Your fast
   lane past most of our process — see the green lane in
   [CONTRIBUTING.md](CONTRIBUTING.md#which-lane-is-your-contribution).
3. **CI reproduction** — the suite runs your test where everyone can see it.
4. **Pasted local logs** — required as provenance on every functional change
   (e.g. `pixi run test tests/...`), but the weakest evidence on its own.

New implementations additionally name their reference (PyTorch, OpenCV,
scikit-image, a paper) in the PR description, so reviewers can check the algorithm
is real and not hallucinated.

### Rule 2 — Don't reinvent kornia

AI tools love writing helpers that already exist. Search `kornia` first and use the
existing utility; if you genuinely can't, say why in the PR. Writing a new
`def warp_affine...` when kornia already has one is grounds for immediate rejection.

### Rule 3 — Own it, and be able to defend it

If a reviewer asks how a function works, you can walk them through it — the math, the
shapes, the edge cases, the design decisions. This is what being the author means
here: not that you typed every character, but that you **own** every line.

Humane interpretation, both directions: written answers are fine, imperfect English
is fine, "let me check and get back to you" is fine. Reviewers commit to asking
targeted questions — about algorithms, shapes, edge cases, trade-offs — not to
running oral exams. What ends a review is disengagement or unresolved substance:
"that's what the AI wrote" as a final answer, or a correctness concern that never
gets addressed. The bar is resolved concerns, not eloquence.

A special mention for comments: hallucinated or redundant comments ("this returns the
input tensor", comments explaining code that was deleted) are the fastest tell of an
unreviewed PR, and they trigger a request for a full manual rewrite.

### Rule 4 — Tell us how it was made

Fill in the AI disclosure in the PR template honestly:

- 🟢 **Human-written** — no AI involved.
- 🟡 **AI-assisted** — AI helped (autocomplete, refactoring, drafts); you reviewed and
  tested every line.
- 🔴 **AI-generated** — an agent produced most of the code or the PR.

**None of these is a bad answer — and the 🟡/🔴 boundary is fuzzy, which is fine.**
Autocomplete, agent edits, generated tests, and human restructuring form a continuum
with no objective line; pick the closest label and add a sentence of detail if
unsure. We will never close a PR over an arguable classification, and we have no
interest in authorship-taxonomy debates — authorship was never the quality gate.
What we sanction is **demonstrable deception**: verification that never ran,
fabricated logs, denying AI use against plain evidence. A 🔴 PR with thorough
verification is welcome here — some of our own most heavily reviewed PRs are exactly
that. An honest 🔴 beats a dishonest 🟡 every time.

## What we do on our side

Policies that only demand things are no fun, so here is our half of the deal:

- **We gate on evidence, not permission.** Self-verifying contributions (test-first bug
  fixes, verified docs fixes, `help wanted` issues with acceptance criteria, benchmark
  results) need no assignment and no waiting — see
  [CONTRIBUTING.md](CONTRIBUTING.md#which-lane-is-your-contribution).
- **We verify before we reject.** Review findings — including those from our AI
  reviewers — are checked by execution, not vibes. Automated closures only ever happen
  on objective criteria (missing test logs, missing issue link where one is required),
  never on a bot's opinion of your code quality.
- **Maintainers: same evidence bar, different governance authority — stated
  honestly.** Maintainer PRs — agent-built ones included — carry pasted execution
  logs, honest disclosure, and review like everyone else's. What maintainers do
  *not* do is ask themselves for permission: they hold the scope-setting authority
  the discuss-first gate exists to route to, so the lane/issue guardrails skip them
  (the validation workflow says so in code). Requiring maintainers to grant
  themselves an issue would be theater; letting them skip the evidence bar would be
  corrosion. We do the first openly and refuse the second — hold us to it.

## This policy is an experiment

AI changed the economics of contribution; this policy is our current best response,
not scripture. We will revisit it after a full contribution season against real
metrics — green-lane merge rate, disclosure disputes, duplicate work on
first-PR-wins issues, time to first review, maintainer load — and change what the
data says is wrong.

## Instructions for AI Reviewers (Copilot / CodeRabbit)

If you are an AI agent (GitHub Copilot, CodeRabbit, etc.) reviewing a PR for Kornia,
you must follow the repository's dedicated reviewer instructions.

The **canonical and up-to-date instructions for AI reviewers** are maintained in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md). That document
defines:

- The expected reviewer persona and responsibilities
- The checks to perform on PR descriptions, code, tests, and comments
- The required enforcement of the rules defined in this `AI_POLICY.md`

Any other document (including this one) should treat `copilot-instructions.md` as the
single source of truth for AI reviewer behaviour. When updating reviewer logic, update
`copilot-instructions.md` first and, if needed, adjust references here.

## Additional Resources

For comprehensive guidance on contributing to Kornia, including development workflows,
code quality standards, testing practices, and AI-assisted development best practices,
see the [Best Practices section](CONTRIBUTING.md#best-practices) in `CONTRIBUTING.md`.
