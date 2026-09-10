#!/usr/bin/env python3
"""
Bulk-download open-access PDFs for every paper referenced in a spreadsheet.

Reads a TSV with a pubmed_id column (and optionally a title column used for
naming), dedupes by PMID, and downloads each paper's PDF from the same
open-access sources as fetch_paper.py (Europe PMC, then Unpaywall) via
paper_sources.py. No paywall bypassing -- a paper with no legal OA copy is
just recorded as such for manual follow-up (see manual_needed.tsv).

Caches progress in <outdir>/.manifest.json so re-running the script:
  - never re-downloads a paper that already succeeded (and still exists on disk)
  - by default retries papers that previously found no OA copy or errored,
    since those are cheap lookups rather than large downloads -- pass
    --skip-attempted once you're done chasing stragglers to stop retrying them.

Each paper is processed independently inside its own try/except, so one
paper's failure never aborts the run. Ctrl-C flushes the manifest before
exiting, and progress is also checkpointed to disk every 10 papers.

Usage:
    python batch_fetch_papers.py hivmut_mutagenesis_combined.tsv -o papers
    python batch_fetch_papers.py hivmut_mutagenesis_combined.tsv -o papers --limit 50
    python batch_fetch_papers.py hivmut_mutagenesis_combined.tsv -o papers --skip-attempted
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error

from paper_sources import (
    build_filename,
    download_pdf,
    find_oa_pdf_candidates,
    resolve_ids_batch,
)

MANIFEST_NAME = ".manifest.json"
TRANSIENT_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError)


def load_rows(tsv_path, id_column, title_column):
    papers = {}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pmid = (row.get(id_column) or "").strip()
            if not pmid or not pmid.isdigit():
                continue
            if pmid not in papers:
                papers[pmid] = {"title": (row.get(title_column) or "").strip() or None}
    return papers


def load_manifest(path):
    if os.path.exists(path):
        with open(path) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_manifest(path, manifest):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def process_one(pmid, ids, title, outdir, email, retries, retry_delay):
    pmcid = ids.get("pmcid")
    doi = ids.get("doi")
    filename = build_filename(pmid=pmid, pmcid=pmcid, doi=doi, title=title)
    dest_path = os.path.join(outdir, filename)

    last_error = None
    for attempt in range(retries + 1):
        try:
            candidates, landing_url = find_oa_pdf_candidates(pmid=pmid, pmcid=pmcid, doi=doi, email=email)
            for pdf_url in candidates:
                try:
                    if download_pdf(pdf_url, dest_path):
                        return {"status": "downloaded", "file": filename, "pmcid": pmcid, "doi": doi}
                except TRANSIENT_ERRORS:
                    continue
            if pmcid:
                # Free to read on PMC, just not in Europe PMC's licensed-for-
                # redistribution OA subset -- needs a manual browser save
                # past NCBI's bot check, not institutional/proxy login.
                status = "pmc_manual"
            else:
                status = "no_oa"
            return {"status": status, "pmcid": pmcid, "doi": doi, "landing_url": landing_url}
        except TRANSIENT_ERRORS as e:
            last_error = str(e)
            if attempt < retries:
                time.sleep(retry_delay * (attempt + 1))
                continue
        except Exception as e:
            return {"status": "error", "error": str(e), "pmcid": pmcid, "doi": doi}
    return {"status": "error", "error": last_error or "unknown transient failure", "pmcid": pmcid, "doi": doi}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tsv", help="TSV file with a pubmed_id column")
    parser.add_argument("-o", "--outdir", required=True, help="Folder to save PDFs into")
    parser.add_argument("--id-column", default="pubmed_id")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--email", default=os.environ.get("UNPAYWALL_EMAIL", "smf252@scarletmail.rutgers.edu"),
                         help="Contact email for the Unpaywall API")
    parser.add_argument("--delay", type=float, default=0.34,
                         help="Seconds to wait between papers (politeness rate limit)")
    parser.add_argument("--retries", type=int, default=2, help="Retries per paper on transient network errors")
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending papers (for testing)")
    parser.add_argument("--skip-attempted", action="store_true",
                         help="Also skip papers already marked no_oa/error in a previous run (default: retry them)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    manifest_path = os.path.join(args.outdir, MANIFEST_NAME)
    manifest = load_manifest(manifest_path)

    papers = load_rows(args.tsv, args.id_column, args.title_column)
    print(f"{len(papers)} unique papers in {args.tsv}")

    pending = []
    for pmid, info in papers.items():
        entry = manifest.get(pmid)
        if entry and entry.get("status") == "downloaded":
            if entry.get("file") and os.path.isfile(os.path.join(args.outdir, entry["file"])):
                continue
        if entry and args.skip_attempted and entry.get("status") in ("no_oa", "pmc_manual", "error"):
            continue
        pending.append(pmid)

    print(f"{len(pending)} papers pending (not yet downloaded)")
    if args.limit:
        pending = pending[: args.limit]
        print(f"Limiting to first {len(pending)} for this run")

    if not pending:
        print("Nothing to do.")
        return

    print("Resolving PMIDs to PMCID/DOI via NCBI ID Converter (batched)...")
    id_map = resolve_ids_batch(pending)

    counts = {"downloaded": 0, "pmc_manual": 0, "no_oa": 0, "error": 0}
    try:
        for i, pmid in enumerate(pending, 1):
            title = papers[pmid]["title"]
            ids = id_map.get(pmid, {})
            result = process_one(pmid, ids, title, args.outdir, args.email, args.retries, args.retry_delay)
            result["title"] = title
            result["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            manifest[pmid] = result

            status = result["status"]
            counts[status] = counts.get(status, 0) + 1
            print(f"[{i}/{len(pending)}] PMID{pmid}: {status}", flush=True)

            if i % 10 == 0:
                save_manifest(manifest_path, manifest)
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted -- saving progress.")
    finally:
        save_manifest(manifest_path, manifest)

    manual_path = os.path.join(args.outdir, "manual_needed.tsv")
    with open(manual_path, "w") as f:
        f.write("pubmed_id\ttitle\tstatus\taction_needed\tlanding_url_or_error\n")
        for pmid, entry in sorted(manifest.items(), key=lambda kv: int(kv[0])):
            status = entry.get("status")
            if status == "downloaded":
                continue
            action = {
                "pmc_manual": "free on PMC -- save from browser, no login needed",
                "no_oa": "no free copy found -- try Rutgers proxy login",
                "error": "check error",
            }.get(status, "")
            detail = entry.get("landing_url") or entry.get("error") or ""
            title = papers.get(pmid, {}).get("title", "")
            f.write(f"{pmid}\t{title}\t{status}\t{action}\t{detail}\n")

    print(
        f"\nDone this run: {counts['downloaded']} downloaded, "
        f"{counts['pmc_manual']} free on PMC but need manual save, "
        f"{counts['no_oa']} no free copy found, {counts['error']} errored."
    )
    total_downloaded = sum(1 for e in manifest.values() if e.get("status") == "downloaded")
    print(f"Total downloaded so far (all runs): {total_downloaded}/{len(papers)}")
    print(f"Papers needing manual retrieval: {manual_path}")
    print("Re-run this script any time to pick up where it left off.")


if __name__ == "__main__":
    main()
