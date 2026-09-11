#!/usr/bin/env python3
"""Build a local HTML page of proxied article links for manually working
through the papers batch_fetch_papers.py couldn't get automatically (no_oa
and/or pmc_manual). Open the generated file in the real, logged-in browser
(see fetch_via_proxy.py's docstring for the X-forwarding setup) and click
through it -- each link routes through the institutional proxy straight to
the article, using whatever session is already authenticated in that
browser tab.

Only includes PMIDs that already have a DOI on file (manifest's own doi
field) -- run resolve_missing_dois.py first to backfill via OpenAlex for
ones NCBI's ID Converter didn't have.

Usage:
    python build_link_queue.py -o ../../data/link_queue.html
    python build_link_queue.py --status no_oa -o ../../data/link_queue_no_oa.html
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
FETCH_MANIFEST = PAPERS_DIR / ".manifest.json"
DEFAULT_PROXY_PREFIX = "https://go.openathens.net/redirector/nyu.edu?url="

PAGE_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Paper fetch queue</title>
<style>
  body {{ font: 14px/1.4 system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
  h1 {{ font-size: 1.2em; }}
  .stats {{ color: #555; margin-bottom: 1em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #ddd; vertical-align: top; }}
  th {{ position: sticky; top: 0; background: #fff; }}
  tr.done {{ opacity: 0.35; }}
  .status {{ font-size: 0.85em; color: #888; }}
  .pmid {{ font-family: monospace; font-size: 0.85em; color: #888; white-space: nowrap; }}
  a.open {{ white-space: nowrap; }}
</style>
<h1>Paper fetch queue</h1>
<p class="stats"><span id="remaining"></span> remaining of {total} &mdash;
  check a row off after you've saved that PDF to <code>data/papers/</code>
  (progress is remembered per-browser via localStorage, not synced anywhere).</p>
<table>
<thead><tr><th></th><th>PMID</th><th>Title</th><th>Status</th><th>Open</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<script>
  const key = p => "paperq_" + p;
  document.querySelectorAll("tr[data-pmid]").forEach(tr => {{
    const pmid = tr.dataset.pmid;
    const cb = tr.querySelector("input[type=checkbox]");
    if (localStorage.getItem(key(pmid)) === "1") {{ cb.checked = true; tr.classList.add("done"); }}
    cb.addEventListener("change", () => {{
      localStorage.setItem(key(pmid), cb.checked ? "1" : "0");
      tr.classList.toggle("done", cb.checked);
      updateCount();
    }});
  }});
  function updateCount() {{
    const rows = document.querySelectorAll("tr[data-pmid]");
    const done = document.querySelectorAll("tr.done").length;
    document.getElementById("remaining").textContent = (rows.length - done);
  }}
  updateCount();
</script>
"""

ROW_TEMPLATE = (
    '<tr data-pmid="{pmid}"><td><input type="checkbox"></td>'
    '<td class="pmid">{pmid}</td><td>{title}</td><td class="status">{status}</td>'
    '<td class="open"><a class="open" href="{link}" target="_blank" rel="noopener">open &rarr;</a></td></tr>'
)


def already_have_pdf(pmid: str) -> bool:
    return any(PAPERS_DIR.glob(f"PMID{pmid}_*.pdf")) or (PAPERS_DIR / f"PMID{pmid}.pdf").exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", type=Path, default=PROJECT_ROOT / "data" / "link_queue.html")
    parser.add_argument("--status", default="no_oa,pmc_manual",
                         help="comma-separated fetch-manifest statuses to include (default: no_oa,pmc_manual)")
    parser.add_argument("--proxy-prefix", default=DEFAULT_PROXY_PREFIX)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    statuses = set(args.status.split(","))
    if not FETCH_MANIFEST.exists():
        sys.exit(f"missing {FETCH_MANIFEST}")
    manifest = json.loads(FETCH_MANIFEST.read_text())

    entries = []
    for pmid, v in manifest.items():
        if v.get("status") not in statuses:
            continue
        if already_have_pdf(pmid):
            continue
        doi = v.get("doi")
        if not doi:
            continue
        entries.append((pmid, v.get("title") or "", v.get("status"), doi))

    # no_oa first (proxy is the *only* automated route left for those);
    # pmc_manual second (proxy is a redundant alternate to PMC's block for these)
    entries.sort(key=lambda e: (e[2] != "no_oa", e[0]))
    if args.limit:
        entries = entries[:args.limit]

    rows = []
    for pmid, title, status, doi in entries:
        link = args.proxy_prefix + f"https://doi.org/{doi}"
        rows.append(ROW_TEMPLATE.format(
            pmid=pmid, title=html.escape(title), status=status, link=html.escape(link),
        ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE_TEMPLATE.format(rows="\n".join(rows), total=len(entries)))
    skipped = sum(1 for pmid, v in manifest.items()
                  if v.get("status") in statuses and not v.get("doi") and not already_have_pdf(pmid))
    print(f"{len(entries)} link(s) -> {args.out}", file=sys.stderr)
    if skipped:
        print(f"({skipped} more matched status but have no DOI on file yet -- "
              f"resolve those via OpenAlex and re-run to add them)", file=sys.stderr)


if __name__ == "__main__":
    main()
