#!/usr/bin/env python3
"""Backfill DOIs into data/papers/.manifest.json for PMIDs that have none on
file, via OpenAlex (see paper_sources.resolve_doi_openalex -- NCBI's own ID
Converter misses a meaningful fraction that OpenAlex still resolves).

Doesn't fetch anything, just fills in the doi field so other tools
(build_link_queue.py, fetch_via_proxy.py, a re-run of batch_fetch_papers.py)
have more to work with. Safe to interrupt -- checkpoints periodically.

Usage:
    python resolve_missing_dois.py --status no_oa,pmc_manual
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from paper_sources import resolve_doi_openalex  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
FETCH_MANIFEST = PAPERS_DIR / ".manifest.json"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def already_have_pdf(pmid: str) -> bool:
    return any(PAPERS_DIR.glob(f"PMID{pmid}_*.pdf")) or (PAPERS_DIR / f"PMID{pmid}.pdf").exists()


def save_manifest(manifest: dict):
    tmp = FETCH_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    tmp.replace(FETCH_MANIFEST)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", default="no_oa,pmc_manual")
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    statuses = set(args.status.split(","))
    manifest = json.loads(FETCH_MANIFEST.read_text())
    pending = [pmid for pmid, v in manifest.items()
               if v.get("status") in statuses and not v.get("doi") and not already_have_pdf(pmid)]
    if args.limit:
        pending = pending[:args.limit]
    log(f"{len(pending)} PMID(s) missing a DOI")

    found = 0
    for i, pmid in enumerate(pending, 1):
        doi = resolve_doi_openalex(pmid)
        if doi:
            manifest[pmid]["doi"] = doi
            found += 1
        if i % 25 == 0 or i == len(pending):
            log(f"  progress: {i}/{len(pending)} ({found} resolved)")
            save_manifest(manifest)
        time.sleep(args.delay)

    save_manifest(manifest)
    log(f"Done. {found}/{len(pending)} resolved.")


if __name__ == "__main__":
    main()
