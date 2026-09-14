# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Collect kornia's dependents from GitHub's dependency graph into docs/source/_data/dependents.json.

GitHub shows "Used by" on the repository page and lists every dependent at
``github.com/kornia/kornia/network/dependents``, but exposes neither through the REST or GraphQL
API. This script walks that HTML listing -- one page of 30 per request, cursor-paginated -- for
repositories and for packages, and stores the exact totals plus the most-starred entries.

Run it by hand and commit the refreshed JSON -- for instance before a release; there is deliberately
no scheduled job for it. ``generate_adoption.py`` renders the file into the Adoption page at
docs-build time::

    python docs/fetch_dependents.py                 # full crawl (~330 requests, ~10 min)
    python docs/fetch_dependents.py --max-pages 5   # quick refresh of the totals

A partial crawl (``--max-pages`` or a network failure) merges into the existing file, so the totals
are always current and the ranking only ever gets more complete.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "source" / "_data" / "dependents.json"

DEPENDENTS_URL = "https://github.com/kornia/kornia/network/dependents"
USER_AGENT = "kornia-docs-dependents/1.0 (+https://github.com/kornia/kornia)"
KEEP_TOP = 300  # entries stored per kind, ranked by stars
DELAY_S = 1.0  # between requests; GitHub serves this page without auth but rate-limits bursts

_COUNT_RE = {
    "repositories": re.compile(r">\s*([\d,]+)\s*Repositor(?:y|ies)\s*<"),  # codespell:ignore repositor
    "packages": re.compile(r">\s*([\d,]+)\s*Packages?\s*<"),
}
_ROW_SPLIT_RE = re.compile(r'<div class="Box-row[^"]*"')
_REPO_RE = re.compile(r'data-hovercard-type="repository"[^>]*href="/([^/"]+/[^/"]+)"')
_NUMBER_RE = re.compile(r"</svg>\s*([\d,]+)\s*</span>")
_NEXT_RE = re.compile(r'href="[^"]*dependents_after=([A-Za-z0-9_=-]+)[^"]*"[^>]*>\s*Next\s*<')


def _get(url: str, retries: int = 5) -> str:
    """Fetch a page, backing off on rate limiting and transient server errors."""
    if not url.startswith("https://github.com/"):
        raise ValueError(f"refusing to fetch {url!r}: only GitHub HTTPS pages are crawled")
    delay = 30.0
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 - scheme checked above
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = float(exc.headers.get("Retry-After") or delay)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            wait = delay
        print(f"  {url}: retrying in {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)
        delay *= 2
    raise RuntimeError("unreachable")


def parse_page(page: str) -> tuple[dict[str, int], list[dict], str | None]:
    """Return (totals, rows, next_cursor) from one dependents page."""
    totals = {}
    for kind, pattern in _COUNT_RE.items():
        match = pattern.search(page)
        if match:
            totals[kind] = int(match.group(1).replace(",", ""))

    rows = []
    for chunk in _ROW_SPLIT_RE.split(page)[1:]:
        repo = _REPO_RE.search(chunk)
        if not repo:
            continue
        numbers = [int(n.replace(",", "")) for n in _NUMBER_RE.findall(chunk)[:2]]
        stars, forks = (numbers + [0, 0])[:2]
        rows.append({"name": html.unescape(repo.group(1)), "stars": stars, "forks": forks})

    next_match = _NEXT_RE.search(page)
    return totals, rows, next_match.group(1) if next_match else None


def crawl(kind: str, max_pages: int) -> tuple[int | None, list[dict], bool]:
    """Walk one listing. Returns (total, rows, complete); a network failure ends the walk early."""
    dependent_type = "REPOSITORY" if kind == "repositories" else "PACKAGE"
    url = f"{DEPENDENTS_URL}?dependent_type={dependent_type}"
    total, rows, cursor, pages = None, [], None, 0
    while True:
        page_url = url if cursor is None else f"{url}&dependents_after={cursor}"
        try:
            totals, page_rows, cursor = parse_page(_get(page_url))
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  {kind}: stopping after {pages} pages: {exc!r}", file=sys.stderr)
            return total, rows, False
        total = totals.get(kind, total)
        rows.extend(page_rows)
        pages += 1
        if pages % 50 == 0:
            print(f"  {kind}: {pages} pages, {len(rows)} rows", file=sys.stderr)
        if cursor is None:
            return total, rows, True
        if max_pages and pages >= max_pages:
            return total, rows, False
        time.sleep(DELAY_S)


def load_existing(path: Path = OUT) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def merge(existing: dict, kind: str, total: int | None, rows: list[dict], complete: bool) -> dict:
    """Replace the stored ranking after a complete crawl; extend it after a partial one."""
    previous = existing.get(kind, {})
    items = {} if complete else {item["name"]: item for item in previous.get("items", [])}
    for row in rows:
        items[row["name"]] = row
    ranked = sorted(items.values(), key=lambda item: (-item["stars"], item["name"].lower()))
    # GitHub's listing stops paginating well before its own total (about 9,700 of 17,500 repositories
    # in 2026), so ``complete`` means "walked to the end of what GitHub serves", and ``listed`` says
    # how many rows that was.
    listed = len(rows) if complete else max(previous.get("listed", 0), len(rows))
    return {
        "count": total if total is not None else previous.get("count", 0),
        "items": ranked[:KEEP_TOP],
        "complete": complete,
        "listed": listed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-pages", type=int, default=0, help="stop each listing after N pages (0 = all)")
    parser.add_argument("--kind", choices=["repositories", "packages"], action="append", help="limit to one listing")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    data = load_existing(args.out)
    for kind in args.kind or ["repositories", "packages"]:
        print(f"crawling {kind}...", file=sys.stderr)
        total, rows, complete = crawl(kind, args.max_pages)
        if total is None and not rows:
            continue  # nothing new; keep the previous entry untouched
        data[kind] = merge(data, kind, total, rows, complete)
        print(
            f"  {kind}: {data[kind]['count']:,} total, {len(rows)} rows fetched, complete={complete}",
            file=sys.stderr,
        )

    data["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%d")
    data["source"] = DEPENDENTS_URL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
