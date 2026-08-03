#!/usr/bin/env python3
"""Check that links in the org's markdown documents still resolve.

Written after an audit found several dead pointers in Kornia's docs that
nobody had noticed:

  * a Code of Conduct routing reports to edgar.riba@arraiy.com, an address
    at a company that no longer operates
  * a link to bubbaloop.kornia.org, a domain that does not resolve
  * a footer link to news.html, a file that was never committed

Relative links are checked against the filesystem and fail the run.
External links are checked over the network, and only genuine deaths
(404/410, DNS failure) fail; hosts that merely refuse robots are reported
as unverified, because a 403 from bot protection says nothing about
whether the page exists.

Email addresses are reported for human review: whether a mailbox is
monitored cannot be determined from here, and that is exactly how the
arraiy address survived for years.
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 25
UA = "Mozilla/5.0 (compatible; kornia-link-check/1.0; +https://www.kornia.org/)"

INCONCLUSIVE = {202, 401, 403, 405, 429, 503, 999}
SKIP_DIRS = {".git", "node_modules", ".venv", "target", "__pycache__"}

# [text](url) and bare <url>
LINK_RE = re.compile(r"\[[^\]]*\]\(\s*<?([^)\s>]+)>?\s*(?:\"[^\"]*\")?\s*\)")
AUTOLINK_RE = re.compile(r"<((?:https?://|mailto:)[^>\s]+)>")

errors: list[str] = []
notes: list[str] = []

# Things that look like an address but are not a mailbox.
NOT_MAILBOXES = re.compile(
    r"@(?:\d+x)?\.(?:png|jpe?g|svg|gif|webp|ico)$|^git@", re.I
)

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
FENCE_RE = re.compile(r"^```.*?^```", re.S | re.M)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def strip_noncontent(text: str) -> str:
    """Drop HTML comments and code so template snippets are not treated as links.

    kornia/kornia's BACKERS.md documents how to add a sponsor logo with an
    example `![Name](./logos/name.svg)` inside an HTML comment. That file
    does not exist and is not meant to; flagging it would be a false alarm.
    """
    text = COMMENT_RE.sub("", text)
    text = FENCE_RE.sub("", text)
    return INLINE_CODE_RE.sub("", text)


def markdown_files() -> list[Path]:
    out: list[Path] = []
    for p in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


def probe(url: str) -> tuple[str, object]:
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return "ok", resp.status
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            return "dead", exc.code
        return "unverified", exc.code
    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        if "Name or service not known" in reason or "nodename nor servname" in reason:
            return "dead", "DNS"
        return "unverified", reason[:60]
    except Exception as exc:  # noqa: BLE001
        return "unverified", type(exc).__name__


def main() -> int:
    # --offline skips every network probe, so pull requests get a fast,
    # deterministic check. The networked run is scheduled, where a
    # third-party blip costs a red cron job rather than a blocked PR.
    offline = "--offline" in sys.argv

    files = markdown_files()
    if not files:
        print("no markdown files found")
        return 0

    external: dict[str, list[str]] = {}
    emails: dict[str, list[str]] = {}

    for path in files:
        rel = path.relative_to(ROOT)
        text = strip_noncontent(path.read_text(encoding="utf-8"))

        targets = set(LINK_RE.findall(text)) | set(AUTOLINK_RE.findall(text))

        for target in targets:
            target = target.strip()
            if not target:
                continue

            if target.startswith("mailto:"):
                emails.setdefault(target[7:], []).append(str(rel))
                continue

            parsed = urlparse(target)
            if parsed.scheme in {"http", "https"}:
                external.setdefault(target, []).append(str(rel))
                continue
            if parsed.scheme:
                continue  # tel:, data:, etc.
            if target.startswith("#"):
                continue  # in-document anchor; heading slugs are not worth guessing

            local = (path.parent / parsed.path).resolve()
            if not local.exists():
                errors.append(f"{rel}: relative link '{target}' does not exist on disk")

    # Bare email addresses outside markdown link syntax, e.g. in prose.
    for path in files:
        rel = path.relative_to(ROOT)
        # The trailing group must not be [\w.-]+, or a sentence-ending period
        # gets swallowed into the address ("hello@kornia.org." != the mailbox).
        addr_re = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
        body = strip_noncontent(path.read_text(encoding="utf-8"))
        for addr in set(re.findall(addr_re, body)):
            if NOT_MAILBOXES.search(addr):
                continue  # image filename or an SSH remote, not an address
            emails.setdefault(addr, []).append(str(rel))

    mode = "offline" if offline else "networked"
    print(f"{len(files)} markdown file(s), {len(external)} external link(s), "
          f"{len(emails)} email address(es)  [{mode}]\n")

    for url in (() if offline else sorted(external)):
        verdict, status = probe(url)
        where = ", ".join(sorted(set(external[url])))
        if verdict == "dead":
            errors.append(f"{where}: DEAD ({status}) {url}")
            print(f"DEAD        {status}  {url}")
        elif verdict == "unverified":
            notes.append(f"{where}: unverified ({status}) {url}")
            print(f"unverified  {status}  {url}")

    if emails:
        print("\nEmail addresses referenced (verify a human still reads these):")
        for addr in sorted(emails):
            domain = addr.split("@")[-1]
            verdict, status = ("skipped", "-") if offline else probe(f"https://{domain}")
            flag = "  <-- domain does not resolve" if (verdict, status) == ("dead", "DNS") else ""
            if flag:
                errors.append(f"{', '.join(sorted(set(emails[addr])))}: "
                              f"email domain {domain} does not resolve ({addr})")
            print(f"  {addr:<40} {', '.join(sorted(set(emails[addr])))}{flag}")

    if notes:
        print(f"\n{len(notes)} unverified (host refused an automated request; not proof of breakage)")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")
        return 1

    print("\nno dead links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
