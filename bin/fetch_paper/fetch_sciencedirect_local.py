#!/usr/bin/env python3
"""
Stand-alone downloader for ScienceDirect / Elsevier articles using Elsevier Developer API.
Run this script on your local machine while connected to your institutional VPN.

Requirements:
    Only Python 3 standard library (no external dependencies needed: uses urllib.request).

Usage:
    # 1. Export your API key or pass it via --api-key:
    export EDP_KEY="your_elsevier_api_key_here"

    # 2. Run the script:
    python3 fetch_sciencedirect_local.py --manifest data/papers/.manifest.json --outdir data/papers/

    # Or test on just 5 papers:
    python3 fetch_sciencedirect_local.py --manifest data/papers/.manifest.json --outdir data/papers/ --limit 5
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (compatible; ScienceDirectFetcher/1.0)"


def sanitize_filename(name, fallback="paper"):
    if not name:
        name = fallback
    name = re.sub(r"[^\w\s.-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:150] or fallback


def build_filename(pmid=None, title=None):
    id_prefix = f"PMID{pmid}" if pmid else "paper"
    name_part = sanitize_filename(title, "") if title else ""
    return f"{id_prefix}_{name_part}.pdf" if name_part else f"{id_prefix}.pdf"


def is_sciencedirect(doi: str) -> bool:
    if not doi:
        return False
    doi_lower = doi.lower()
    return any(p in doi_lower for p in ("10.1016/", "10.1006/"))


def fetch_pdf_elsevier(doi: str, api_key: str, dest_path: str, inst_token: str = None, timeout: int = 45):
    """
    Fetch full-text PDF via Elsevier Article Retrieval API.
    Returns (success: bool, msg: str)
    """
    clean_doi = doi.strip()
    # URL quote the DOI path component properly
    encoded_doi = urllib.parse.quote(clean_doi, safe="")
    url = f"https://api.elsevier.com/content/article/doi/{encoded_doi}"

    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/pdf",
        "User-Agent": USER_AGENT,
    }
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read()

        if not body.startswith(b"%PDF") and "pdf" not in content_type.lower():
            # Sometimes API returns JSON or XML error with 200 OK
            preview = body[:200].decode("utf-8", errors="replace")
            return False, f"Response not a PDF ({content_type}): {preview}"

        # Write to temp file then atomic rename
        tmp_path = dest_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(body)
        os.replace(tmp_path, dest_path)
        return True, f"{len(body) / 1024:.1f} KB"

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:200]
        return False, f"HTTP {e.code}: {err_body}"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Fetch ScienceDirect PDFs via Elsevier API on local/VPN machine.")
    parser.add_argument("--manifest", default="data/papers/.manifest.json", help="Path to manifest JSON")
    parser.add_argument("--outdir", default="data/papers", help="Directory to save downloaded PDFs")
    parser.add_argument("--api-key", default=os.environ.get("EDP_KEY"), help="Elsevier API key (or set EDP_KEY env var)")
    parser.add_argument("--inst-token", default=os.environ.get("EDP_INSTTOKEN"), help="Optional Elsevier InstToken")
    parser.add_argument("--limit", type=int, default=None, help="Max papers to fetch")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between requests (rate limit safe)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Elsevier API key must be provided via --api-key or EDP_KEY environment variable.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.manifest):
        print(f"ERROR: Manifest file '{args.manifest}' not found.")
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Filter papers: ScienceDirect DOIs that do not already have a valid PDF on disk
    pending = []
    for pmid, entry in manifest.items():
        doi = entry.get("doi") or ""
        if not is_sciencedirect(doi):
            continue

        title = entry.get("title")
        filename = build_filename(pmid=pmid, title=title)
        dest_path = os.path.join(args.outdir, filename)

        # Check if file already exists with nonzero size
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
            continue

        pending.append({
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "dest_path": dest_path,
            "filename": filename
        })

    print(f"Found {len(pending)} ScienceDirect paper(s) needing download.")
    if args.limit:
        pending = pending[:args.limit]
        print(f"Limiting to first {len(pending)} paper(s).")

    if not pending:
        print("Nothing to download. All ScienceDirect PDFs already present!")
        return

    success_count = 0
    fail_count = 0
    updated_manifest = False

    try:
        for idx, item in enumerate(pending, 1):
            pmid = item["pmid"]
            doi = item["doi"]
            dest = item["dest_path"]

            print(f"[{idx}/{len(pending)}] PMID:{pmid} ({doi})... ", end="", flush=True)
            ok, msg = fetch_pdf_elsevier(doi, args.api_key, dest, inst_token=args.inst_token)

            if ok:
                success_count += 1
                manifest[pmid]["status"] = "downloaded"
                manifest[pmid]["file"] = item["filename"]
                manifest[pmid]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                updated_manifest = True
                print(f"SUCCESS ({msg})")
            else:
                fail_count += 1
                print(f"FAILED ({msg})")

            time.sleep(args.delay)

            # Checkpoint manifest every 25 downloads
            if updated_manifest and success_count % 25 == 0:
                tmp = args.manifest + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=1, sort_keys=True)
                os.replace(tmp, args.manifest)
                updated_manifest = False

    except KeyboardInterrupt:
        print("\nInterrupted by user! Saving progress...")
    finally:
        if updated_manifest:
            tmp = args.manifest + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=1, sort_keys=True)
            os.replace(tmp, args.manifest)
            print("Manifest successfully updated.")

    print(f"\nDone! Successes: {success_count}, Failures: {fail_count}")


if __name__ == "__main__":
    main()
