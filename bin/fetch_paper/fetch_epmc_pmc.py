#!/usr/bin/env python3
"""
Fetch papers that have a PMCID using Europe PMC's open PDF endpoint:
  https://europepmc.org/api/getPdf?pmcid=<PMCID>

SAFE CONCURRENCY DESIGN:
  Because `batch_fetch_papers.py` may be actively running and rewriting
  `.manifest.json`, this script:
  1. Only READS `.manifest.json` (as a snapshot to discover PMCIDs and titles).
  2. Saves downloaded PDFs directly to `data/papers/PMID<pmid>_<title>.pdf`.
  3. Checks `os.path.exists()` on the target PDF before downloading so it never
     duplicates work or overwrites existing files.
  4. Records its own progress in `.epmc_pmc_manifest.json` so there is ZERO
     file contention or lock clash with `batch_fetch_papers.py`.
  5. Once `batch_fetch_papers.py` completes, this script (or a simple sync step)
     can merge `.epmc_pmc_manifest.json` into `.manifest.json`.

Usage:
    python fetch_epmc_pmc.py
    python fetch_epmc_pmc.py --limit 100
    python fetch_epmc_pmc.py --delay 0.5
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
MAIN_MANIFEST = PAPERS_DIR / ".manifest.json"
EPMC_MANIFEST = PAPERS_DIR / ".epmc_pmc_manifest.json"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
EPMC_GET_PDF_URL = "https://europepmc.org/api/getPdf?pmcid={pmcid}"


def sanitize_filename(name, fallback):
    if not name:
        name = fallback
    import re
    name = re.sub(r"[^\w\s.-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:150] or fallback


def build_filename(pmid=None, pmcid=None, title=None):
    if pmid:
        id_prefix = f"PMID{pmid}"
    elif pmcid:
        id_prefix = pmcid
    else:
        id_prefix = "paper"
    name_part = sanitize_filename(title, "") if title else ""
    return f"{id_prefix}_{name_part}.pdf" if name_part else f"{id_prefix}.pdf"


def get_existing_pmids() -> set[str]:
    if not PAPERS_DIR.exists():
        return set()
    import re
    existing = set()
    for p in PAPERS_DIR.glob("*.pdf"):
        if p.stat().st_size > 1000:
            m = re.match(r"PMID(\d+)", p.name)
            if m:
                existing.add(m.group(1))
    return existing


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_json(path: Path, data: dict):
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(path)
    except Exception:
        try:
            path.write_text(json.dumps(data, indent=1, sort_keys=True))
        except Exception:
            pass


def download_epmc_pdf(pmcid: str, dest_path: Path, timeout: int = 30) -> bool:
    url = EPMC_GET_PDF_URL.format(pmcid=pmcid)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            body = resp.read()
            if body.startswith(b"%PDF") or "pdf" in content_type:
                dest_path.write_bytes(body)
                return True
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delay", type=float, default=0.35, help="Polite delay between requests (default: 0.35s)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers to attempt")
    parser.add_argument("--merge-into-main", action="store_true", help="Merge downloaded status into main .manifest.json")
    parser.add_argument("--retry-failed", action="store_true", help="Retry papers previously marked not_available")
    args = parser.parse_args()

    if not PAPERS_DIR.exists():
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    main_manifest = load_json(MAIN_MANIFEST)
    epmc_manifest = load_json(EPMC_MANIFEST)

    if args.merge_into_main:
        print("Merging .epmc_pmc_manifest.json into .manifest.json...")
        merged_count = 0
        for pmid, info in epmc_manifest.items():
            if info.get("status") == "downloaded":
                if pmid in main_manifest:
                    main_manifest[pmid]["status"] = "downloaded"
                    main_manifest[pmid]["file"] = info.get("file")
                    main_manifest[pmid]["updated"] = info.get("updated")
                    merged_count += 1
        save_json(MAIN_MANIFEST, main_manifest)
        print(f"Successfully merged {merged_count} downloaded records into {MAIN_MANIFEST}")
        return

    # Filter targets: papers that have a PMCID and aren't downloaded yet
    existing_pmids = get_existing_pmids()
    targets = []
    for pmid, entry in main_manifest.items():
        pmcid = entry.get("pmcid")
        if not pmcid:
            continue
        title = entry.get("title")
        dest_filename = build_filename(pmid=pmid, pmcid=pmcid, title=title)
        dest_path = PAPERS_DIR / dest_filename

        if pmid in existing_pmids or dest_path.exists():
            continue
        if epmc_manifest.get(pmid, {}).get("status") == "downloaded":
            continue
        if not args.retry_failed and epmc_manifest.get(pmid, {}).get("status") == "not_available":
            continue

        targets.append({
            "pmid": pmid,
            "pmcid": pmcid,
            "title": title,
            "dest_filename": dest_filename,
            "dest_path": dest_path
        })

    print(f"Found {len(targets)} candidate papers with PMCID pending download.")
    if args.limit:
        targets = targets[:args.limit]
        print(f"Limiting to first {len(targets)} papers.")

    if not targets:
        print("No papers pending download.")
        return

    success_count = 0
    fail_count = 0

    try:
        for idx, item in enumerate(targets, 1):
            pmid = item["pmid"]
            pmcid = item["pmcid"]
            dest_path = item["dest_path"]
            filename = item["dest_filename"]

            ok = download_epmc_pdf(pmcid, dest_path)
            if ok:
                success_count += 1
                epmc_manifest[pmid] = {
                    "status": "downloaded",
                    "pmcid": pmcid,
                    "file": filename,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S")
                }
                print(f"[{idx}/{len(targets)}] PMID:{pmid} ({pmcid}) -> SUCCESS ({filename})", flush=True)
            else:
                fail_count += 1
                epmc_manifest[pmid] = {
                    "status": "not_available",
                    "pmcid": pmcid,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S")
                }
                print(f"[{idx}/{len(targets)}] PMID:{pmid} ({pmcid}) -> not available on EPMC", flush=True)

            if idx % 10 == 0:
                save_json(EPMC_MANIFEST, epmc_manifest)

            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
    finally:
        save_json(EPMC_MANIFEST, epmc_manifest)
        # Auto-merge downloaded items into main manifest on exit/interrupt/error
        try:
            cur_main = load_json(MAIN_MANIFEST)
            merged = 0
            for pmid, info in epmc_manifest.items():
                if info.get("status") == "downloaded" and pmid in cur_main:
                    if cur_main[pmid].get("status") != "downloaded":
                        cur_main[pmid]["status"] = "downloaded"
                        cur_main[pmid]["file"] = info.get("file")
                        cur_main[pmid]["updated"] = info.get("updated")
                        merged += 1
            if merged > 0:
                save_json(MAIN_MANIFEST, cur_main)
                print(f"Automatically merged {merged} newly downloaded records into {MAIN_MANIFEST}")
        except Exception as e:
            print(f"Warning: could not auto-merge into main manifest: {e}")

    print(f"\nDone! Downloaded: {success_count}, Not available: {fail_count}")
    print(f"Recorded results in {EPMC_MANIFEST}")


if __name__ == "__main__":
    main()
