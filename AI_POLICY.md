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

### Rule 1 — Show us it runs

Code that "looks correct" is not enough, no matter who or what wrote it.

- Every PR with functional changes includes a pasted snippet of your local test run
  (e.g. `pixi run test tests/...`).
- Bug fixes should arrive **test-first**: a test that fails on `main` and passes with
  your fix is the single most convincing thing a PR can contain. It is also your fast
  lane past most of our process — see the green lane in
  [CONTRIBUTING.md](CONTRIBUTING.md#which-lane-is-your-contribution).
- New implementations name their reference (PyTorch, OpenCV, scikit-image, a paper) in
  the PR description, so reviewers can check the algorithm is real and not hallucinated.

### Rule 2 — Don't reinvent kornia

AI tools love writing helpers that already exist. Search `kornia` first and use the
existing utility; if you genuinely can't, say why in the PR. Writing a new
`def warp_affine...` when kornia already has one is grounds for immediate rejection.

### Rule 3 — Be able to explain it

If a reviewer asks how a function works, you can walk them through it — the math, the
shapes, the edge cases. Answering "that's what the AI wrote" ends the review. This is
what being the author means here: not that you typed every character, but that you
**own** every line and every design decision.

A special mention for comments: hallucinated or redundant comments ("this returns the
input tensor", comments explaining code that was deleted) are the fastest tell of an
unreviewed PR, and they trigger a request for a full manual rewrite.

### Rule 4 — Tell us how it was made

Fill in the AI disclosure in the PR template honestly:

- 🟢 **Human-written** — no AI involved.
- 🟡 **AI-assisted** — AI helped (autocomplete, refactoring, drafts); you reviewed and
  tested every line.
- 🔴 **AI-generated** — an agent produced most of the code or the PR.

**None of these is a bad answer.** A 🔴 PR with thorough verification and a human who
can explain every line is welcome here — some of our own most heavily reviewed PRs are
exactly that. What closes PRs is **mislabeling**, or "verification" that turns out to
be imaginary. An honest 🔴 beats a dishonest 🟡 every time.

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
- **We hold ourselves to the same standard.** Maintainer PRs — agent-built ones
  included — carry pasted execution logs and go through review like everyone else's.
  You are welcome to hold us to this.

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
