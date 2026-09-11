#!/usr/bin/env python3
"""Build data/manifest.tsv: one row per unique PMID referenced in the hivmut
mutagenesis data, joining the paper-fetch status with pipeline progress.

Sources:
  data/papers/.manifest.json   fetch status per PMID (downloaded / pmc_manual
                                / no_oa / error), written by
                                bin/fetch_paper/batch_fetch_papers.py
  data/papers/*.pdf            which PDFs actually exist on disk -- checked
                                directly rather than trusting the fetch
                                manifest's "downloaded" status, since a PDF
                                can also arrive by manual save (the
                                pmc_manual/no_oa workflow) without the fetch
                                tool ever being told
  data/extracted/<stem>/       extraction + per-model answer + critic flag
                                files, written by bin/batch_process.py and
                                bin/critic_flag.py

Output columns:
  pmid, title, fetch_status, pmcid, doi, pdf_file, extracted,
  qwen_answered, gptoss_answered, flagged

flagged is a count 0-2: how many of the two models' critic verdicts (see
critic_flag.py) were YES -- 0 means neither, 2 means both agreed.

Usage:
    python build_manifest.py [-o data/manifest.tsv]
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
EXTRACTED_ROOT = PROJECT_ROOT / "data" / "extracted"
FETCH_MANIFEST = PAPERS_DIR / ".manifest.json"
HIVMUT_TSV = PROJECT_ROOT / "data" / "hivmut" / "hivmut_mutagenesis_combined.tsv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "manifest.tsv"

PMID_RE = re.compile(r"^PMID(\d+)(?:_.*)?$")


def load_fetch_manifest() -> dict:
    if not FETCH_MANIFEST.exists():
        sys.exit(f"missing {FETCH_MANIFEST} -- extract it from the papers zip first")
    return json.loads(FETCH_MANIFEST.read_text())


def load_tsv_titles() -> dict:
    """Fallback titles for any PMID not in the fetch manifest (shouldn't
    normally happen, but the hivmut TSV is the ground truth for which PMIDs
    matter)."""
    titles = {}
    if not HIVMUT_TSV.exists():
        return titles
    with open(HIVMUT_TSV, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pmid = (row.get("pubmed_id") or "").strip()
            if pmid.isdigit():
                titles.setdefault(pmid, (row.get("title") or "").strip() or None)
    return titles


def pdf_stems_by_pmid() -> dict:
    """PMID -> pdf stem, from what's actually on disk in data/papers/."""
    stems = {}
    for pdf in PAPERS_DIR.glob("*.pdf"):
        m = PMID_RE.match(pdf.stem)
        if m:
            stems[m.group(1)] = pdf.stem
    return stems


def read_flag(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            return line.upper()
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    fetch_manifest = load_fetch_manifest()
    fallback_titles = load_tsv_titles()
    stems_by_pmid = pdf_stems_by_pmid()

    all_pmids = set(fetch_manifest) | set(fallback_titles)

    rows = []
    counts = {
        "fetch_status": {}, "extracted": 0, "qwen_answered": 0,
        "gptoss_answered": 0, "flagged": 0,
    }
    for pmid in sorted(all_pmids, key=int):
        entry = fetch_manifest.get(pmid, {})
        title = entry.get("title") or fallback_titles.get(pmid) or ""
        fetch_status = entry.get("status", "not_attempted")
        counts["fetch_status"][fetch_status] = counts["fetch_status"].get(fetch_status, 0) + 1

        stem = stems_by_pmid.get(pmid)
        pdf_file = f"{stem}.pdf" if stem else ""

        extracted = qwen_answered = gptoss_answered = False
        qwen_flag = gptoss_flag = ""
        if stem:
            out_dir = EXTRACTED_ROOT / stem
            extracted = (out_dir / f"{stem}.txt").exists()
            qwen_answered = (out_dir / f"{stem}.qwen.llm").exists()
            gptoss_answered = (out_dir / f"{stem}.gptoss.llm").exists()
            qwen_flag = read_flag(out_dir / f"{stem}.qwen.flag")
            gptoss_flag = read_flag(out_dir / f"{stem}.gptoss.flag")

        flagged = (qwen_flag == "YES") + (gptoss_flag == "YES")

        if extracted:
            counts["extracted"] += 1
        if qwen_answered:
            counts["qwen_answered"] += 1
        if gptoss_answered:
            counts["gptoss_answered"] += 1
        if flagged:
            counts["flagged"] += 1

        rows.append({
            "pmid": pmid,
            "title": title,
            "fetch_status": fetch_status,
            "pmcid": entry.get("pmcid") or "",
            "doi": entry.get("doi") or "",
            "pdf_file": pdf_file,
            "extracted": extracted,
            "qwen_answered": qwen_answered,
            "gptoss_answered": gptoss_answered,
            "flagged": flagged,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} papers -> {args.out}", file=sys.stderr)
    print(f"fetch status: {counts['fetch_status']}", file=sys.stderr)
    print(f"extracted: {counts['extracted']}, qwen answered: {counts['qwen_answered']}, "
          f"gptoss answered: {counts['gptoss_answered']}", file=sys.stderr)
    print(f"flagged (critic said YES on qwen's or gpt-oss's answer): {counts['flagged']}", file=sys.stderr)


if __name__ == "__main__":
    main()
